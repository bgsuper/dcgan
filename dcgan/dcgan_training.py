import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

def training(loss, var_list, learning_rate=0.001):
    global_step = tf.Variable(0, name='global_step', trainable=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


    with tf.control_dependencies(update_ops):
        if FLAGS.model == 'dcgan_wasserstein':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
        train_op = optimizer.minimize(loss=loss, var_list=var_list, global_step=global_step)

    return train_op, global_step