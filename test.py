from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
import tensorflow as tf
import numpy as np
from imageio import imread, imwrite
import random
import os, sys, shutil
import pickle
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
tf.flags.DEFINE_string("test_dataset", "strongly_labeled_test.txt", 
        "Path to test dataset file")
tf.flags.DEFINE_string("store", "StPaul", 
        "Store for which to compute features (if mode == compute_features)")

# Hyperparameters
tf.flags.DEFINE_integer("batch_size", "10", 
        "Batch size for training")
tf.flags.DEFINE_integer("num_preds", "10", 
        "Number of test predictions")

# Misc
tf.flags.DEFINE_string('mode', "visualize", 
        "Mode accuracy/visualize")
tf.flags.DEFINE_bool('debug', "True", 
        "Debug mode: True/ False")

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224
ORIGINAL_IMAGE_SIZE = [720, 1280, 3]


def _loss_function(logits_list, labels_list,
                   pre_features, post_features, mid_features, weights):
    entropy_loss = tf.add_n([tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels) for logits,labels in zip(logits_list, labels_list)])
    tf.summary.scalar("entropy", entropy_loss)

    # check broadcasting semantics
    import ipdb; ipdb.set_trace()
    mse_loss = tf.losses.mean_squared_error(labels=mid_features,
                                            predictions=weights * pre_features + (1-weights) * post_features)
    tf.summary.scalar("mse", mse_loss)

    loss = entropy_loss + mse_loss
    tf.summary.scalar("loss", loss)

    return loss


def main(argv):
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")

    if FLAGS.mode == 'accuracy':

        #####################
        # Dataset pipeline
        #####################

        def _parse_function(filename, llabel, rlabel):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_image(image_string)
            image_decoded.set_shape(ORIGINAL_IMAGE_SIZE)
            image_resized = tf.image.resize_images(image_decoded, [IMAGE_SIZE, IMAGE_SIZE])
            return image_resized, llabel, rlabel

        with open('label_map.txt') as f:
            label_map = { lbl.strip(): i for i,lbl in enumerate(f.readlines()) }
        num_cats = len(label_map)

        print(FLAGS.test_dataset)
        with open(FLAGS.test_dataset) as f:
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

        iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
        batch = iterator.get_next()

        test_init_op = iterator.make_initializer(test_dataset)

        images, llabels, rlabels = batch

        _, _, lpreds, _, rpreds, _ = inference(images, num_cats, keep_probability, FLAGS.model_dir, FLAGS.fcn_dir, FLAGS.debug)

        sess = tf.Session()

        print("Setting up Saver...", flush=True)
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        # ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        # print(ckpt.model_checkpoint_path)
        print(FLAGS.logs_dir)
        saver.restore(sess, FLAGS.logs_dir)
        print("Model restored...", flush=True)

        sess.run(test_init_op)
        lbls, preds = [], []
        while True:
            try:
                ll_, lp_, rl_, rp_ = sess.run([llabels, lpreds, rlabels, rpreds], feed_dict={keep_probability: 1.0})
                lbls.append(ll_)
                preds.append(lp_)
                lbls.append(rl_)
                preds.append(rp_)
            except tf.errors.OutOfRangeError:
                lbls = np.concatenate(lbls)
                preds = np.concatenate(preds)
                acc = np.mean(lbls == preds)
                print("Accuracy: %g" % (acc))
                break


    elif FLAGS.mode == 'compute_features':
        with open('label_map.txt') as f:
            label_map = { i: lbl.strip() for i,lbl in enumerate(f.readlines()) }
        num_cats = len(label_map)

        from glob import glob
        store_dir = '/mnt/grocery_data/Traderjoe/{}'.format(FLAGS.store)
        imgfiles = sorted(glob('{}/image/*.jpg'.format(store_dir)))

        def _parse_function(filename):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_image(image_string)
            image_decoded.set_shape(ORIGINAL_IMAGE_SIZE)
            image_resized = tf.image.resize_images(image_decoded, [IMAGE_SIZE, IMAGE_SIZE])
            return image_resized

        imgfiles = tf.constant(imgfiles)
        test_dataset = tf.data.Dataset.from_tensor_slices(imgfiles)
        test_dataset = test_dataset.map(_parse_function)
        test_dataset = test_dataset.batch(FLAGS.batch_size)

        iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
        test_init_op = iterator.make_initializer(test_dataset)
        images = iterator.get_next()

        img_ftrs, _, _, _, _, _ = inference(images, num_cats, keep_probability, FLAGS.model_dir, FLAGS.fcn_dir, FLAGS.debug)

        sess = tf.Session()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...", flush=True)

        features = []
        sess.run(test_init_op)
        i = 0
        while True:
            try:
                if i % 10 == 0:
                    print('step {}'.format(i))
                ftrs = sess.run(img_ftrs, feed_dict={keep_probability: 1.0})
                features.append(ftrs)
                i += 1
            except tf.errors.OutOfRangeError:
                break
        features = np.concatenate(features)
        pickle.dump(features, open('{}/features.pkl'.format(store_dir), 'wb'))

    else:
        images = tf.placeholder(tf.float32, shape=([None] + ORIGINAL_IMAGE_SIZE), name="input_image")

        num_preds = FLAGS.num_preds
        imgs = []
        with open('/mnt/grocery_data/Traderjoe/StPaul/strongly_labeled_test.txt') as f:
            lines = f.readlines()
            lines = random.sample(lines, num_preds)
            for i, line in enumerate(lines):
                imgfile, _ = line.strip().split(' ')
                shutil.copy(imgfile, 'preds/img_{}.jpg'.format(i))
                img = np.expand_dims(imread(imgfile), 0)
                imgs.append(img)
        # store_dir = '/mnt/grocery_data/Traderjoe/Shoreview/image'
        # img_nums = random.sample(range(1,23197), num_preds)
        # for i, img_num in enumerate(img_nums):
        #     # imgfile = '{}/image{:07d}.jpg'.format(store_dir, img_num)
        #     # shutil.copy(imgfile, 'preds/img_{}.jpg'.format(i))
        #     # img = np.expand_dims(imread(imgfile), 0)
        #     # imgs.append(img)
        imgs = np.concatenate(imgs)

        with open('/mnt/grocery_data/Traderjoe/StPaul/labels.txt') as f:
            label_map = { i: lbl.strip() for i,lbl in enumerate(f.readlines()) }
        num_cats = len(label_map)

        resized_images = tf.image.resize_images(images, [IMAGE_SIZE, IMAGE_SIZE])
        _, _, lpreds, _, rpreds, _ = inference(resized_images, num_cats, keep_probability, FLAGS.model_dir, FLAGS.fcn_dir, FLAGS.debug)

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, 'logs_classify_fast_decay/model.ckpt-16')
        print("Model restored...", flush=True)

        lpreds, rpreds = sess.run([lpreds, rpreds], feed_dict={images: imgs, keep_probability: 1.0})

        for i in range(num_preds):
            print('{}\t\t{}'.format(label_map[lpreds[i]], label_map[rpreds[i]]))


if __name__ == '__main__':
    tf.app.run()
