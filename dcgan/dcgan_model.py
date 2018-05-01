import tensorflow as tf
from dcgan.dcgan_layers import *


FLAGS = tf.app.flags.FLAGS


def generator(generator_input, is_training, keep_prob, reuse=False):

    with tf.variable_scope('generator', reuse=reuse) as scope:

        # transform the input to a 4*4, 1024 channels feature map
        last_hidden_channel = FLAGS.last_hidden_channel
        filter_size = FLAGS.filter_size  # default =4
        print('gen shape')
        print(generator_input.get_shape())
        gen_h1_ = tf.layers.conv2d_transpose(generator_input, last_hidden_channel, [filter_size, filter_size],
                                             strides=(2,2), padding='valid')
        gen_h1 = leaky_relu(tf.layers.batch_normalization(gen_h1_, training=is_training), 0.2)

        # deconv layer2
        gen_h2_ = tf.layers.conv2d_transpose(gen_h1, last_hidden_channel//2, [filter_size, filter_size],
                                             strides=(2, 2), padding='same')
        gen_h2 = leaky_relu(tf.layers.batch_normalization(gen_h2_, training=is_training), 0.2)

        # deconv layer3
        gen_h3_ = tf.layers.conv2d_transpose(gen_h2, last_hidden_channel//4, [filter_size, filter_size],
                                             strides=(2, 2), padding='same')
        gen_h3 = leaky_relu(tf.layers.batch_normalization(gen_h3_, training=is_training), 0.2)

        # deconv layer4
        gen_h4_ = tf.layers.conv2d_transpose(gen_h3, last_hidden_channel//8, [filter_size, filter_size],
                                             strides=(2, 2), padding='same')
        gen_h4 = leaky_relu(tf.layers.batch_normalization(gen_h4_, training=is_training), 0.2)

        # output_layer
        gen_output_ = tf.layers.conv2d_transpose(gen_h4, 1, [filter_size, filter_size],
                                             strides=(1, 1), padding='same')
        gen_output = tf.nn.tanh(tf.layers.batch_normalization(gen_output_, training=is_training))
        print('gen output')
        print(gen_output.get_shape())
    return gen_output


def discriminator(discriminator_input, is_training, keep_prob, reuse=False):
    last_hidden_channel = FLAGS.last_hidden_channel # default = 1024
    filter_size = FLAGS.filter_size  # default =4
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        # 1. conv
        print('disc input size')
        print(discriminator_input.get_shape())
        disc_hidden1_ = tf.layers.conv2d(discriminator_input, last_hidden_channel//8, [filter_size, filter_size],
                                         strides=(2,2), padding='same', name='disc_h1')
        disc_hidden1 = leaky_relu(tf.layers.batch_normalization(disc_hidden1_, training=is_training), 0.2)
        # 2. conv
        disc_hidden2_ = tf.layers.conv2d(disc_hidden1, last_hidden_channel // 4, [filter_size, filter_size],
                                         strides=(2,2), padding='same', name='disc_h2')
        disc_hidden2 = leaky_relu(tf.layers.batch_normalization(disc_hidden2_, training=is_training), 0.2)
        # 3. conv
        disc_hidden3_ = tf.layers.conv2d(disc_hidden2, last_hidden_channel // 2, [filter_size, filter_size],
                                         strides=(2,2), padding='same', name='disc_h3')
        disc_hidden3 = leaky_relu(tf.layers.batch_normalization(disc_hidden3_, training=is_training), 0.2)
        # 4. conv
        disc_hidden4_ = tf.layers.conv2d(disc_hidden3, last_hidden_channel, [filter_size, filter_size],
                                         strides=(2,2), padding='same', name='disc_h4')
        # disc_hidden4_dropout = tf.layers.dropout(disc_hidden4_, rate=keep_prob)
        disc_hidden4 = leaky_relu(tf.layers.batch_normalization(disc_hidden4_, training=is_training), 0.2)


        # FC layer
        print(disc_hidden4.get_shape().as_list())
        hidden_4_height, hiden_4_width = disc_hidden4.get_shape().as_list()[1], disc_hidden4.get_shape().as_list()[2]
        disc_output_ = tf.layers.conv2d(disc_hidden4, 1, [hidden_4_height, hiden_4_width],
                                        strides=(1,1), padding='valid', name='disc_output')
        # disc_output_dropout = tf.layers.dropout(disc_output_, rate=keep_prob)

        disc_output = tf.nn.sigmoid(disc_output_)

    return disc_output, disc_output_


def dcgan(generator_input, images_real, training_generator, keep_prob, reuse=True):
    generated_images = generator(generator_input, is_training=True, reuse=True, keep_prob=keep_prob)

    if training_generator:
        return discriminator(
            generated_images,
            is_training=True,
            keep_prob=keep_prob,
            reuse=reuse)
    else:
        images = [images_real, generated_images]
        return discriminator(
            images,
            is_training=True,
            keep_prob=keep_prob,
            reuse=reuse)


