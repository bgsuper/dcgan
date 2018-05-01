import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS


def loss_cal(logits, labels, name):
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name=name))
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
