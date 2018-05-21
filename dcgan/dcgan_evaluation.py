import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS


def loss_cal(out, labels, name):
    if FLAGS.model == 'dcgan_wasserstein':
        loss = tf.reduce_mean(out)
    else:
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=labels, name=name))

    return loss


# def discriminator_loss(discriminator_logits):
#     pass
#
#
# def generator_loss(generator_logits):
#     pass
#
#
# def dcgan_loss(discriminator_logits, generator_logits):
#     return discriminator_loss(discriminator_logits) + generator_loss(generator_logits)
