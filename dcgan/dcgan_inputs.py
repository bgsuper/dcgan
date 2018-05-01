import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

import threading
import skimage
import skimage.io

FLAGS = tf.app.flags.FLAGS


def get_filename_list(path):

    image_filenames = sorted(os.listdir(os.path.join(path, 'images')))
    # Adding correct path to lists
    step = 0
    for name in image_filenames:
        image_filenames[step] = os.path.join(path+ 'images' + name)
        step = step+1

    return  image_filenames


def dataset_reader(image_filename_queue):
    # get png encoded image
    imageValue = tf.read_file(image_filename_queue)

    # decodes a png image into uint8 or unit16 tensor
    # returns a tensor of type dtype with shape [height, width, depth]

    image_bytes = tf.image.decode_png(imageValue)
    image = tf.reshape(image_bytes, (FLAGS.image_h, FLAGS.image_w, FLAGS.image_c))

    return image


def dataset_inputs(image_filenames, batch_size, shuffle=True):


    images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)

    image_filename_queue = tf.train.slice_input_producer([images], shuffle=shuffle)

    image = dataset_reader(image_filename_queue)
    reshaped_image = tf.cast(image, tf.float32)
    min_fraction_of_examples_in_queue = FLAGS.fraction_of_examples_in_queue
    min_queue_examples = int(FLAGS.num_examples_epoch_train*
                                 min_fraction_of_examples_in_queue)

    print('Filling queue with %d input images before starting to train. '
          'This may take some time.' % min_queue_examples)

    # generate a batch of images
    return _generate_image_batch(reshaped_image, min_queue_examples, batch_size, shuffle=shuffle)


def _generate_image_batch(image, min_queue_examples,
                          batch_size, shuffle=True):
    num_preprocess_threads = 1

    if shuffle:
        images = tf.train.shuffle_batch([image],
                                        batch_size=batch_size,
                                        num_threads=num_preprocess_threads,
                                        capacity=min_queue_examples + 1*batch_size,  # 1 =  number of channels in images
                                        min_after_dequeue=min_queue_examples)
    else:
        images = tf.train.batch([image],
                                batch_size=batch_size,
                                num_threads=num_preprocess_threads,
                                capacity=min_queue_examples + batch_size)

    tf.summary.image('training_images', images)
    print('generating image and label batch')
    return images


def get_all_test_data(im_list):
    images = []

    for im_filename in im_list:
        im = np.array(skimage.io.imread(im_filename), np.float32)
        im = im[np.newaxis]

        images.append(im)

    return images

def data_input_mnist(batch_size):

    mnist_data = input_data.read_data_sets(FLAGS.train_dir, one_hot=True, reshape=[], dtype=dtypes.float32)
    image = tf.image.resize_images(mnist_data.train.images, [FLAGS.image_h,FLAGS.image_w])
    image = (image - 0.5)/0.5
    reshaped_image = tf.reshape(tf.train.slice_input_producer([image]), shape= [FLAGS.image_h,FLAGS.image_w,1])
    min_fraction_of_examples_in_queue = FLAGS.fraction_of_examples_in_queue
    min_queue_examples = int(FLAGS.num_examples_epoch_train *
                             min_fraction_of_examples_in_queue)


    return tf.train.shuffle_batch([reshaped_image], batch_size=batch_size,
                                        capacity=min_queue_examples + 1*batch_size,  # 1 =  number of channels in images
                                        min_after_dequeue=min_queue_examples)


def placeholder_inputs(batch_size):
    images_real = tf.placeholder(tf.float32, shape=[batch_size, FLAGS.image_h, FLAGS.image_w, 1], name='real_images')
    labels_real = tf.placeholder(tf.float32, shape=[batch_size, 1,1,1], name='real_labels')
    images_fake = tf.placeholder(tf.float32, shape=[batch_size, FLAGS.image_h, FLAGS.image_w, 1], name='fake_images')
    labels_fake = tf.placeholder(tf.float32, shape=[batch_size, 1,1,1], name='fake_labels')
    generator_input = tf.placeholder(tf.float32, shape=[batch_size,1,1, FLAGS.latent_dim], name='generator_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_probability')
    is_training = tf.placeholder(tf.bool, name='is_trainging')
    reuse = tf.placeholder(tf.bool, name='reuse')
    return images_real, labels_real, images_fake, labels_fake, generator_input, keep_prob, is_training, reuse


