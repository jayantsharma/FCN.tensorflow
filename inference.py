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

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'


def vgg_net(weights, image, debug):
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
            if debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net, restore_vars


def inference(image, num_cats, keep_prob, model_dir, fcn_dir, debug):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...", flush=True)
    model_data = utils.get_model_data(model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net, restore_vars = vgg_net(weights, processed_image, debug)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        restore_vars += [W6, b6]

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if debug:
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

        model_path = tf.train.latest_checkpoint(fcn_dir)
        restore_fn = slim.assign_from_checkpoint_fn(model_path, restore_vars, ignore_missing_vars=False)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     restore_fn(sess)
        # import ipdb; ipdb.set_trace()
    return ftrs_1, llogits, lpreds, rlogits, rpreds, restore_fn
