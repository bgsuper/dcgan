from .dcgan_model import  discriminator, generator, dcgan
from .dcgan_training import training
from .utils import *
from .dcgan_inputs import placeholder_inputs, dataset_inputs, get_filename_list, get_all_test_data,data_input_mnist
from .dcgan_evaluation import loss_cal

import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

"""AFFECTS HOW CODE RUNS"""

tf.app.flags.DEFINE_string('model', 'dcgan_mnist',
                           """ Defining what version of the model to run """)
tf.app.flags.DEFINE_string('model', 'dcgan_wasserstein',
                           """ Run the wasserstein version of the model """)

#Model specific parameters
tf.app.flags.DEFINE_integer('latent_dim', "100",
                            """ dimension of the latent variable which is used for generating fake images""")
tf.app.flags.DEFINE_integer('last_hidden_channel', '1024',
                            """ number of channels applying convolution to the latent vector """)
tf.app.flags.DEFINE_integer('filter_size', '4', """size of conv filters""")


#Training
tf.app.flags.DEFINE_string('log_dir',"./tmp/dcgan_mnist", #Training is default on, unless testing or finetuning is set to "True"
                           """ dir to store training ckpt """)
tf.app.flags.DEFINE_integer('max_steps', "40000",
                            """ max_steps for training """)


#Testing
tf.app.flags.DEFINE_boolean('testing', False, #True or False
                            """ Whether to run test or not """)
tf.app.flags.DEFINE_string('model_ckpt_dir', "./tmp/dcgan_mnist/model.ckpt-22500",
                           """ checkpoint file for model to use for testing """)
tf.app.flags.DEFINE_boolean('save_image', True,
                            """ Whether to save predicted image """)
tf.app.flags.DEFINE_string('res_output_dir', "./tmp/dcgan_mnist/result_imgs",
                            """ Directory to save result images when running test """)
#Finetuning
tf.app.flags.DEFINE_boolean('finetune', True, #True or False
                           """ Whether to finetune or not """)
tf.app.flags.DEFINE_string('finetune_dir', 'tmp/dcgan_mnist/model.ckpt-22500',
                           """ Path to the checkpoint file to finetune from """)


""" TRAINING PARAMETERS"""
tf.app.flags.DEFINE_integer('batch_size', "64",
                            """ train batch_size """)
tf.app.flags.DEFINE_integer('test_batch_size', "1",
                            """ batch_size for training """)
tf.app.flags.DEFINE_integer('eval_batch_size', "6",
                            """ Eval batch_size """)



""" DATASET SPECIFIC PARAMETERS """
#Directories
tf.app.flags.DEFINE_string('dataset', 'mnist', """MNIST data set""")
tf.app.flags.DEFINE_string('train_dir', os.path.join('.','data', FLAGS.dataset), """ path to training images """)

#Dataset size. #Epoch = one pass of the whole dataset.
tf.app.flags.DEFINE_integer('num_examples_epoch_train', "50",
                           """ num examples per epoch for train """)
tf.app.flags.DEFINE_integer('num_examples_epoch_test', "80",
                           """ num examples per epoch for test """)
tf.app.flags.DEFINE_integer('num_examples_epoch_val', "50",
                           """ num examples per epoch for test """)
tf.app.flags.DEFINE_float('fraction_of_examples_in_queue', "0.1",
                           """ Fraction of examples from datasat to put in queue. Large datasets need smaller value, otherwise memory gets full. """)


#Image size and classes
tf.app.flags.DEFINE_integer('image_h', "32",
                            """ image height """)
tf.app.flags.DEFINE_integer('image_w', "32",
                            """ image width """)
tf.app.flags.DEFINE_integer('image_c', "1",
                            """ number of image channels (RGB) (the depth) """)
tf.app.flags.DEFINE_integer('num_classes', "2", #classes are "real" and "fake"
                            """ total class number """)

# #FOR TESTING:
# TEST_ITER = FLAGS.num_examples_epoch_test // FLAGS.batch_size


tf.app.flags.DEFINE_float('moving_average_decay', "0.99",#"0.9999", #https://www.tensorflow.org/versions/r0.12/api_docs/python/train/moving_averages
                           """ The decay to use for the moving average""")


if(FLAGS.model == "dcgan_mnist" or FLAGS.model == "basic_dropout"):
    tf.app.flags.DEFINE_string('conv_init', 'xavier', # xavier / var_scale
                            """ Initializer for the convolutional layers. One of: "xavier", "var_scale".  """)
    tf.app.flags.DEFINE_string('optimizer', "SGD",
                            """ Optimizer for training. One of: "adam", "SGD", "momentum", "adagrad". """)

elif(FLAGS.model == "extended" or FLAGS.model == "extended_dropout"):
    tf.app.flags.DEFINE_string('conv_init', 'var_scale', # xavier / var_scale
                            """ Initializer for the convolutional layers. One of "msra", "xavier", "var_scale".  """)
    tf.app.flags.DEFINE_string('optimizer', "adagrad",
                            """ Optimizer for training. One of: "adam", "SGD", "momentum", "adagrad". """)
else:
    raise ValueError("Determine which initalizer you want to use. Non exist for model ", FLAGS.model)
