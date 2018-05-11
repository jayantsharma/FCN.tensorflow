from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
import tensorflow as tf
import numpy as np
from imageio import imread, imwrite
slim = tf.contrib.slim

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS

# Checkpoint params
tf.flags.DEFINE_string("model_dir", "Model_zoo/", 
        "Path to vgg model mat")
tf.flags.DEFINE_string("fcn_dir", "logs_5e5/", 
        "Path to FCN checkpoint")
tf.flags.DEFINE_string("logs_dir", "logs_classify/", 
        "Path to logs directory")

# Hyperparameters
tf.flags.DEFINE_integer("batch_size", "10", 
        "Batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-5", 
        "Learning rate for Adam Optimizer")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5,
    'Number of epochs after which learning rate decays.')

# Misc
tf.flags.DEFINE_string('mode', "train", 
        "Mode train/ test/ visualize")
tf.flags.DEFINE_bool('debug', "True", 
        "Debug mode: True/ False")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224
ORIGINAL_IMAGE_SIZE = [720, 1280, 3]


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    restore_vars = []
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            # Utilizing stored weights of ImageNet pretrained network to provide the correct shapes
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.weight_variable(np.transpose(kernels, (1, 0, 2, 3)).shape, name=name + "_w")
            bias = utils.bias_variable(bias.reshape(-1).shape, name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
            restore_vars += [kernels, bias]
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net, restore_vars


def inference(image, num_cats, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...", flush=True)
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net, restore_vars = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        restore_vars += [W6, b6]

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        restore_vars += [W7, b7]

        shape = relu_dropout7.shape.as_list()
        shp = shape[1] * shape[2] * shape[3]    # 7 * 7 * 4096 = 200704
        downsize = 1000                         # shp // 32                    # 7 * 7 * 128  = 6272
        relu_dropout7 = tf.reshape(relu_dropout7, (-1, shp))
        W9 = utils.weight_variable([shp, downsize], name="W9")
        b9 = utils.bias_variable([downsize], name="b9")
        ftrs_1 = tf.add(tf.matmul(relu_dropout7, W9), b9)

        # downsize2 = downsize // 16        #        # 7 * 7 * 16  = 784
        # W10 = utils.weight_variable([downsize, downsize2], name="W10")
        # b10 = utils.bias_variable([downsize2], name="b10")
        # ftrs_1 = tf.add(tf.matmul(ftrs_0, W10), b10)

        W11 = utils.weight_variable([downsize, num_cats], name="W11")
        b11 = utils.bias_variable([num_cats], name="b11")
        llogits = tf.add(tf.matmul(ftrs_1, W11), b11)
        lpreds = tf.argmax(llogits, axis=1)

        W12 = utils.weight_variable([downsize, num_cats], name="W12")
        b12 = utils.bias_variable([num_cats], name="b12")
        rlogits = tf.add(tf.matmul(ftrs_1, W12), b12)
        rpreds = tf.argmax(rlogits, axis=1)

        ##############################################################################
        # Feature extraction ends here, the rest are just classification layers
        ##############################################################################

        # W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        # b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        # conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
        # restore_vars += [W8, b8]

        # # now to upscale to actual image size
        # deconv_shape1 = image_net["pool4"].get_shape()
        # W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        # b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        # conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        # fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        # restore_vars += [W_t1, b_t1]

        # deconv_shape2 = image_net["pool3"].get_shape()
        # W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        # b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        # conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        # fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
        # restore_vars += [W_t2, b_t2]

        # shape = tf.shape(image)
        # deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        # W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        # b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        # conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        # restore_vars += [W_t3, b_t3]

        model_path = tf.train.latest_checkpoint(FLAGS.fcn_dir)
        restore_fn = slim.assign_from_checkpoint_fn(model_path, restore_vars, ignore_missing_vars=False)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     restore_fn(sess)
        # import ipdb; ipdb.set_trace()
    return llogits, lpreds, rlogits, rpreds, restore_fn

        # annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    # return tf.expand_dims(annotation_pred, dim=3), conv_t3


def _configure_learning_rate(num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
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


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")

    #####################
    # Dataset pipeline
    #####################

    def _parse_function(filename, llabel, rlabel):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_decoded.set_shape(ORIGINAL_IMAGE_SIZE)
        image_resized = tf.image.resize_images(image_decoded, [IMAGE_SIZE, IMAGE_SIZE])
        return image_resized, llabel, rlabel

    ## Train pipeline
    # with np.load("train.npz") as data:
    #     # train_images = data["images"]
    #     # train_llabels = data["llabels"]
    #     # train_rlabels = data["rlabels"]
    with open('/mnt/grocery_data/Traderjoe/StPaul/strongly_labeled_train.txt') as f:
        lines = f.readlines()
        train_imgfiles, train_llabels, train_rlabels = [], [], []
        for line in lines:
            imgfile, labels = line.strip().split(' ')
            train_imgfiles.append(imgfile)
            llabel, rlabel = labels.split(':')
            train_llabels.append(llabel)
            train_rlabels.append(rlabel)
    label_map = { lbl: i for i,lbl in enumerate(set(train_llabels + train_rlabels)) }
    num_cats = len(label_map)
    imgfiles = tf.constant(train_imgfiles)
    llabels = tf.constant([label_map[lbl] for lbl in train_llabels])
    rlabels = tf.constant([label_map[lbl] for lbl in train_rlabels])
    train_dataset = tf.data.Dataset.from_tensor_slices((imgfiles, llabels, rlabels))
    train_dataset = train_dataset.map(_parse_function)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    num_samples_per_epoch = len(train_imgfiles)


    ## Test pipeline
    # with np.load("test.npz") as data:
    #     # test_images = data["images"]
    #     # test_llabels = data["llabels"]
    #     # test_rlabels = data["rlabels"]
    with open('/mnt/grocery_data/Traderjoe/StPaul/strongly_labeled_test.txt') as f:
        lines = f.readlines()
        test_imgfiles, test_llabels, test_rlabels = [], [], []
        for line in lines:
            imgfile, labels = line.strip().split(' ')
            test_imgfiles.append(imgfile)
            llabel, rlabel = labels.split(':')
            test_llabels.append(llabel)
            test_rlabels.append(rlabel)
    imgfiles = tf.constant(test_imgfiles)
    llabels = tf.constant([label_map[lbl] for lbl in test_llabels])
    rlabels = tf.constant([label_map[lbl] for lbl in test_rlabels])
    test_dataset = tf.data.Dataset.from_tensor_slices((imgfiles, llabels, rlabels))
    test_dataset = test_dataset.map(_parse_function)
    test_dataset = test_dataset.batch(FLAGS.batch_size)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    batch = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    images, llabels, rlabels = batch

    llogits, lpreds, rlogits, rpreds, restore_fn = inference(images, num_cats, keep_probability)

    _, laccuracy_op = tf.metrics.accuracy(llabels, lpreds)
    # tf.summary.scalar("laccuracy", laccuracy_op)
    _, raccuracy_op = tf.metrics.accuracy(rlabels, rpreds)
    # tf.summary.scalar("raccuracy", raccuracy_op)

    # tf.summary.image("input_image", image, max_outputs=5)
    # tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=5)
    # tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=5)

    lloss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=llogits,
                                                                          labels=llabels,
                                                                          name="lentropy")))
    tf.summary.scalar("lentropy", lloss)
    rloss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rlogits,
                                                                          labels=rlabels,
                                                                          name="rentropy")))
    tf.summary.scalar("rentropy", rloss)
    loss = lloss + rloss
    tf.summary.scalar("entropy", loss)

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
    # restore FCN weights
    restore_fn(sess)

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...", flush=True)

    sess.run(test_init_op)
    ll, lp, rl, rp = [], [], [], []
    while True:
        try:
            ll_, lp_, rl_, rp_ = sess.run([llabels, lpreds, rlabels, rpreds], feed_dict={keep_probability: 1.0})
            ll.append(ll_)
            lp.append(lp_)
            rl.append(rl_)
            rp.append(rp_)
            # lacc, racc = sess.run([laccuracy_op, raccuracy_op], feed_dict={keep_probability: 1.0})
        except tf.errors.OutOfRangeError:
            ll = np.concatenate(ll)
            lp = np.concatenate(lp)
            rl = np.concatenate(rl)
            rp = np.concatenate(rp)
            lacc = np.mean(ll == lp)
            racc = np.mean(rl == rp)
            print("Initial Accuracy: %g, %g" % (lacc, racc))
            break
    import sys; sys.exit()

    epochs_trained = sess.run(global_step) // num_samples_per_epoch
    if FLAGS.mode == "train":
        for itr in range(epochs_trained, MAX_ITERATION):
            sess.run(train_init_op)
            feed_dict = { keep_probability: .85 }

            while True:
                try:
                    sess.run(train_op, feed_dict=feed_dict)
                    gs = sess.run(global_step)
                    if gs % 10 == 0:
                        ltrain_loss, rtrain_loss, train_loss, summary_str = sess.run([lloss, rloss, loss, summary_op], feed_dict=feed_dict)
                        print("Step: %d, Losses: %g, %g, %g" % (gs, ltrain_loss, rtrain_loss, train_loss), flush=True)
                        summary_writer.add_summary(summary_str, gs)
                except tf.errors.OutOfRangeError:
                    break

            # Now run test routine
            # run till pipeline is exhausted and print accuracy
            # can write a test_op here
            # OPTIONAL add misclassified images to pipeline
            sess.run(test_init_op)
            while True:
                try:
                    lacc, racc = sess.run([laccuracy_op, raccuracy_op], feed_dict={keep_probability: 1.0})
                except tf.errors.OutOfRangeError:
                    print("Epochs: %d, Accuracy: %g, %g" % (itr+1, lacc, racc))
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr+1)
                    break
                # valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                #                                        keep_probability: 1.0})

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr, flush=True)


if __name__ == "__main__":
    tf.app.run()
