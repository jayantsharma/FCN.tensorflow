from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
import tensorflow as tf
import numpy as np
from imageio import imread, imwrite
import random
slim = tf.contrib.slim

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
from inference import vgg_net, inference
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS

# Checkpoint params
tf.flags.DEFINE_string("model_dir", "Model_zoo/", 
        "Path to vgg model mat")
tf.flags.DEFINE_string("fcn_dir", "logs_5e5/", 
        "Path to FCN checkpoint")
tf.flags.DEFINE_string("logs_dir", "logs_complete", 
        "Path to logs directory")
tf.flags.DEFINE_string("initial_ckpt", "logs_classify_fast_decay/model.ckpt-16", 
        "Initial checkpoint path for model trained to classify")

# Hyperparameters
tf.flags.DEFINE_integer("batch_size", "1", 
        "Batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-6", 
        "Learning rate for Adam Optimizer")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2,
    'Number of epochs after which learning rate decays.')

# Misc
tf.flags.DEFINE_string('mode', "train", 
        "Mode train/ test/ visualize")
tf.flags.DEFINE_bool('debug', "False", 
        "Debug mode: True/ False")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224
ORIGINAL_IMAGE_SIZE = [720, 1280, 3]


def _configure_learning_rate(num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    print('Learning rate will decay every: {} steps'.format(decay_steps))
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      .94,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')


def train(loss_val, var_list, global_step, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads, global_step=global_step)


#####################
# Dataset pipeline
#####################
def setup_data_pipeline():

    def _parse_function(marker, filename, *args):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_decoded.set_shape(ORIGINAL_IMAGE_SIZE)
        image_resized = tf.image.resize_images(image_decoded, [IMAGE_SIZE, IMAGE_SIZE])
        return tuple([marker, image_resized] + list(args))

    with open('/mnt/grocery_data/Traderjoe/StPaul/labels.txt') as f:
        label_map = { lbl.strip(): i for i,lbl in enumerate(f.readlines()) }
    num_cats = len(label_map)

    """
    Ideas
    1. zip dataset - im1, lbls1, im2, lbls2, im3, w
    2. shuffle
    3. insert markers - m1, m2, m3 to make sure things work
    4. if shit happens, meditate
    """
    with open('/mnt/grocery_data/Traderjoe/StPaul/unlabeled_data.txt') as f:
        lines = f.readlines()
        im1s, llbls1, rlbls1, im2s, llbls2, rlbls2, im3s, ws = [ [] for _ in range(8) ]
        for line in lines:
            im1, lbls1, im2, lbls2, im3, w = line.strip().split(' ')

            im1s.append(im1)
            im2s.append(im2)
            im3s.append(im3)

            llbl1, rlbl1 = lbls1.split(':')
            llbls1.append(llbl1)
            rlbls1.append(rlbl1)

            llbl2, rlbl2 = lbls2.split(':')
            llbls2.append(llbl2)
            rlbls2.append(rlbl2)

            ws.append(float(w))

    num_samples_per_epoch = len(im1s)
    markers = list(range(len(im1s)))

    imgfiles = tf.constant(im1s)
    llabels = tf.constant([label_map[lbl] for lbl in llbls1])
    rlabels = tf.constant([label_map[lbl] for lbl in rlbls1])
    dataset1 = tf.data.Dataset.from_tensor_slices((markers, imgfiles, llabels, rlabels))
    dataset1 = dataset1.map(_parse_function)

    imgfiles = tf.constant(im2s)
    llabels = tf.constant([label_map[lbl] for lbl in llbls2])
    rlabels = tf.constant([label_map[lbl] for lbl in rlbls2])
    dataset2 = tf.data.Dataset.from_tensor_slices((markers, imgfiles, llabels, rlabels))
    dataset2 = dataset2.map(_parse_function)

    imgfiles = tf.constant(im3s)
    ws = tf.constant(ws)
    dataset3 = tf.data.Dataset.from_tensor_slices((markers, imgfiles, ws))
    dataset3 = dataset3.map(_parse_function)

    train_dataset = tf.data.Dataset.zip((dataset1, dataset2, dataset3))
    train_dataset = train_dataset.shuffle(buffer_size=10000)


    # use only first image for prediction
    with open('/mnt/grocery_data/Traderjoe/StPaul/strongly_labeled_test.txt') as f:
        lines = f.readlines()
        test_imgfiles, test_llabels, test_rlabels, test_weights = [], [], [], []
        for line in lines:
            imgfile, labels = line.strip().split(' ')
            test_imgfiles.append(imgfile)
            llabel, rlabel = labels.split(':')
            test_llabels.append(llabel)
            test_rlabels.append(rlabel)
            test_weights.append(0.)
    markers = list(range(len(test_imgfiles)))
    imgfiles = tf.constant(test_imgfiles)
    llabels = tf.constant([label_map[lbl] for lbl in test_llabels])
    rlabels = tf.constant([label_map[lbl] for lbl in test_rlabels])

    dataset1 = tf.data.Dataset.from_tensor_slices((markers, imgfiles, llabels, rlabels))
    dataset1 = dataset1.map(_parse_function)
    dataset2 = tf.data.Dataset.from_tensor_slices((markers, imgfiles, llabels, rlabels))
    dataset2 = dataset2.map(_parse_function)
    dataset3 = tf.data.Dataset.from_tensor_slices((markers, imgfiles, test_weights))
    dataset3 = dataset3.map(_parse_function)

    test_dataset = tf.data.Dataset.zip((dataset1, dataset2, dataset3))

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    batch = iterator.get_next()
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    return batch, train_init_op, test_init_op, num_samples_per_epoch, num_cats


def _loss_function(logits, labels,
                   features, weight):

    # entropy_loss = tf.add_n([tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels) for logits,labels in zip(logits_list, labels_list)])
    entropy_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
    tf.summary.scalar("entropy", entropy_loss)

    # check broadcasting semantics
    # might need to scale this - scale being a hyperparameter
    mse_loss = tf.norm(features[1,:] - (weight * features[0,:] + (1-weight) * features[2,:])) / 1000
    tf.summary.scalar("mse", mse_loss)

    loss = entropy_loss + mse_loss
    tf.summary.scalar("loss", loss)

    return loss


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")

    batch, train_init_op, test_init_op, num_samples_per_epoch, num_cats = setup_data_pipeline()
    (m1, im1, llbl1, rlbl1), (m2, im2, llbl2, rlbl2), (m3, im3, w) = batch
    images = tf.stack([im1,im2,im3])

    features, llogits, lpreds, rlogits, rpreds, _ = inference(images, num_cats, keep_probability, FLAGS.model_dir, FLAGS.fcn_dir, FLAGS.debug)

    loss = _loss_function(
            tf.concat((llogits[:2,:], rlogits[:2,:]), axis=0),
            [llbl1, llbl2, rlbl1, rlbl2],
            features, w
            )

    global_step = tf.train.get_or_create_global_step()
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)

    learning_rate = _configure_learning_rate(num_samples_per_epoch, global_step)
    tf.summary.scalar("learning_rate", learning_rate)
    train_op = train(loss, trainable_var, global_step, learning_rate)

    print("Setting up summary op...", flush=True)
    summary_op = tf.summary.merge_all()

    sess = tf.Session()

    print("Setting up Saver...", flush=True)
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...", flush=True)
    else:
        saver.restore(sess, FLAGS.initial_ckpt)
        print("Initial checkpoint restored...", flush=True)
        # can reassign global step using tf.assign here
        global_step_var = [v for v in tf.global_variables() if v.name == 'global_step:0'][0]
        sess.run(tf.assign(global_step_var, 0))
        assert sess.run(global_step) == 0


    # test_op, called after each epoch
    def test(step=0):
        sess.run(test_init_op)
        ls, ps = [], []
        while True:
            try:
                l1, l2, p1, p2 = sess.run([llbl1, rlbl1, lpreds, rpreds], feed_dict={keep_probability: 1.0})
                ls += [l1,l2]
                ps += [p1[0], p2[0]]
            except tf.errors.OutOfRangeError:
                ls = np.array(ls)
                ps = np.array(ps)
                acc = np.mean(ls == ps)
                print("Steps: {}, Accuracy: {}".format(step, acc))
                if step > 0:
                    saver.save(sess, FLAGS.logs_dir + "/model.ckpt", step)
                break

    sess.run(train_init_op)
    max_t = 0
    for i in range(sess.run(global_step), MAX_ITERATION):
        if i % 500 == 0:
            # OPTIONAL add misclassified images to pipeline
            # test(i)
            # sess.run(train_init_op)
            saver.save(sess, FLAGS.logs_dir + "/model.ckpt", i)

        t1,t2,t3,_ = sess.run([m1, m2, m3, train_op], feed_dict={ keep_probability: .85 })
        # shuffle nervousness
        assert t1 == t2
        assert t2 == t3
        gs = sess.run(global_step)
        if gs % 10 == 0:
            loss_, summary_str = sess.run([loss, summary_op], feed_dict={ keep_probability: .85 })
            print("Step: %d, Losses: %g" % (gs, loss_), flush=True)
            summary_writer.add_summary(summary_str, gs)


if __name__ == "__main__":
    tf.app.run()
