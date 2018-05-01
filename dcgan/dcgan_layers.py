from __future__ import division, absolute_import, print_function, unicode_literals

import tensorflow as tf
import math
FLAGS = tf.app.flags.FLAGS

# layers
def batch_norm(x, is_training, outpput_shape, scope):


    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(x, is_training=True,
                                                        center=False, decay=FLAGS.moving_average_decay, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(x, is_training=False,
                                                        center=False, reuse=True, decay=FLAGS.moving_average_decay, scope=scope))


def get_weight_initializer():
    if(FLAGS.conv_init == "var_scale"):
        initializer = tf.contrib.layers.variance_scaling_initializer()
    elif(FLAGS.conv_init == "xavier"):
        initializer=tf.contrib.layers.xavier_initializer()
    else:
        raise ValueError("Chosen weight initializer does not exist")
    return initializer


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size)) / float(stride))


def convolution2D_classifier(input_layer,
                             filter_shape,
                             with_initializer = False):

    if with_initializer:
        initializer = get_weight_initializer()
        weights_ = tf.get_variable(name='weights', shape=filter_shape, initializer=initializer)
        bias_ =tf.get_variable(name='bias', shape=[FLAGS.num_classes], initializer=initializer)
    else:
        weights_ = tf.Variable(
            tf.truncated_normal(filter_shape, stddev=0.1), name='weights')

        bias_ = tf.Variable(tf.truncated_normal([FLAGS.num_classes], stddev=0.1), name='bias')

    return tf.add(tf.nn.conv2d(input_layer, weights_, strides=[1,1,1,1], padding='SAME'), bias_)


def convolution2D_bn(input_layer,
                     filter_shape,
                     is_trainging,
                     filter_strides=[1, 1, 1, 1],
                     with_initializer = False,
                     name=None,
                     padding="SAME"):
    # filter_shape should be the form: [filter_height, filter_width, inpout_chanel, output_chanel]
    output_shape = filter_shape[3]

    with tf.variable_scope(name) as scope:
        if with_initializer:
            initializer = get_weight_initializer()
            weights_ = tf.get_variable(name='weights', shape=filter_shape, initializer=initializer)
            bias_ =tf.get_variable(name='bias', shape=[output_shape], initializer=initializer)
        else:
            weights_ = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name='weights')

            bias_ = tf.Variable(tf.truncated_normal([output_shape], stddev=0.1), name='bias')

        conv_ = tf.add(tf.nn.conv2d(input_layer, weights_, filter_strides, padding), bias_)


        batch_normalization_ = batch_norm(conv_, is_trainging, output_shape, scope)

    return batch_normalization_


def convolution2D_bn_reLu(input_layer,
                          filter_shape,
                          is_training,
                          filter_strides=[1, 1, 1, 1],
                          with_inititializer=False,
                          name=None,
                          padding="SAME"):
    bn = convolution2D_bn(input_layer,
                          filter_shape,
                          is_training,
                          filter_strides=filter_strides,
                          with_initializer=with_inititializer,
                          name=name,
                          padding=padding)

    return tf.nn.relu(bn)


def pooling(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool'):
    pool, pool_indices = tf.nn.max_pool_with_argmax(input_layer,
                                                    ksize=ksize,
                                                    strides=strides,
                                                    padding=padding,
                                                    name=name)

    return pool, pool_indices


def unpool_with_argmax(pool, pool_indices, name='unpool', ksize=[1, 2, 2, 1]):
    input_shape = pool.get_shape().as_list()
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

    flat_input_shape = tf.reduce_prod(input_shape)
    flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(pool, [flat_input_shape])

    batch_range = tf.reshape(tf.range(output_shape[0], dtype=pool_indices.dtype), shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(pool_indices) * batch_range
    b = tf.reshape(b, [flat_input_shape, 1])
    flat_indices = tf.reshape(pool_indices, [flat_input_shape, 1])
    flat_indices = tf.concat([b, flat_indices], 1)

    ret = tf.scatter_nd(flat_indices, pool_, shape=flat_output_shape)
    ret = tf.reshape(ret, output_shape)

    return ret


def get_deconv_filter(filter_shape):
    # reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    # it is a bilinear filter
    # filter_shape should be the form: [filter_height, filter_width, inpout_chanel, output_chanel]
    # with filter_height = filter_width

    height = filter_shape[0]
    width = filter_shape[1]

    f = tf.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2 * f)
    bilinear = tf.zeros([height, width])

    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value

    weights_ = tf.zeros(filter_shape)

    for i in range(height):
        weights_[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights_, dtype=tf.float32)

    return tf.get_variable(name='deconv_filter', initializer=init, shape=weights_.shape)


def leaky_relu(input_layer, alpha):
    return tf.nn.relu(input_layer) - alpha*tf.nn.relu(-1*input_layer)

def deconv_layer(input_layer,
                 filter_shape,
                 output_shape,
                 strides=[1, 2, 2, 1],
                 name=None):
    filter_ = get_deconv_filter(filter_shape)

    return tf.nn.conv2d_transpose(input_layer, filter_, output_shape, strides=strides, padding="SAME")
