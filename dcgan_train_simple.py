import os
import tensorflow as tf
import time
from datetime import datetime
import numpy as np
import scipy.misc
import dcgan

FLAGS = tf.app.flags.FLAGS

def train():
    """
    for each epoch:
        * draw random points in the latent space (random noise)
        * Generate images with ' generator' using this random noise
        * Mix the generated image with real ones
        * Train ' discriminator' using these mixed images, with corresponding targets, either "real" or "false"
        * Draw new random points in the latent space
        * Train 'gan' using these random vectors, with targets that all say "these are real images"
    """
    startstep = 0
    if FLAGS.train_dir == os.path.join('.','data', 'mnist'):
        image_filenames = None
    else:
        image_filenames = dcgan.get_filename_list(FLAGS.train_dir)

    with tf.Graph().as_default():
        images_real, labels_real, images_fake, labels_fake, generator_input, keep_prob, is_training, reuse = \
            dcgan.placeholder_inputs(batch_size=FLAGS.batch_size)

        images_real = dcgan.data_input_mnist(FLAGS.batch_size)


        # BUILD GRAPH
        print("Add operators to graph")
        images_generated = dcgan.generator(generator_input, is_training, keep_prob)
        # dcgan.generator(generator_input, is_training=is_training, keep_prob=keep_prob)
        _, discriminator_logits_real = dcgan.discriminator(images_real, is_training=is_training, keep_prob=keep_prob, reuse=False)
        _, generator_logits = dcgan.discriminator(images_generated, is_training=is_training, keep_prob=keep_prob, reuse=True)

        discriminator_logits_fake = generator_logits

        labels_real_ = np.ones([FLAGS.batch_size, 1, 1, 1]).astype('float32')
        labels_fake_ = np.zeros([FLAGS.batch_size, 1, 1, 1]).astype('float32')

        discriminator_loss_real = dcgan.loss_cal(discriminator_logits_real, labels_real_, name="disc_loss_real")
        discriminator_loss_fake = dcgan.loss_cal(discriminator_logits_fake, labels_fake_, name="disc_loss_real")
        discriminator_loss = 0.5*discriminator_loss_real + 0.5*discriminator_loss_fake
        generator_loss = dcgan.loss_cal(generator_logits, labels_real_, name="gen_loss")

        vars_trainable = tf.trainable_variables()
        vars_disctriminator = [var for var in vars_trainable if var.name.startswith('discriminator')]
        vars_generator = [var for var in vars_trainable if var.name.startswith('generator')]

        discriminator_train_op, global_step = dcgan.training(discriminator_loss, vars_disctriminator, learning_rate=0.0000004)
        generator_train_op, global_step = dcgan.training(generator_loss, vars_generator, learning_rate=0.00002)

        # accuracy = tf.argmax()
        summary = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=100000)
        z_fix = np.random.uniform(-1, 1, size = [FLAGS.batch_size,1,1, FLAGS.latent_dim])
        with tf.Session() as sess:
            print("\n =====================================================")
            print("  Training from scratch with model: ", FLAGS.model)
            print("\n    Batch size is: ", FLAGS.batch_size)
            print("    ckpt files are saved to: ", FLAGS.log_dir)
            print("    Max iterations to train is: ", FLAGS.max_steps)
            print(" =====================================================")

            sess.run(tf.variables_initializer(tf.global_variables()))
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, os.path.join(FLAGS.model_ckpt_dir, "dcgan_mnist.ckpt"))
            # start the queue runners
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

            """ start iteration to train the network """
            for step in range(startstep+1, startstep+FLAGS.max_steps+1):
                # run generative model with random inputs

                # train discriminator, fix the parameters of generator
                generator_input_ = np.random.uniform(-1, 1, size = [FLAGS.batch_size,1,1, FLAGS.latent_dim])

                labels_real_ = np.ones([FLAGS.batch_size, 1, 1, 1])
                labels_fake_ = np.zeros([FLAGS.batch_size, 1, 1, 1])

                # package the generated images with the feeded real images and shuffle
                images_batch = sess.run(fetches=images_real)

                train_feed_dict_discriminator = {images_real: images_batch,
                                                 is_training: True,
                                                 generator_input: generator_input_,
                                                 keep_prob: 0.5}

                start_time = time.time()

                _, discriminator_loss_value= \
                    sess.run([discriminator_train_op, discriminator_loss], feed_dict=train_feed_dict_discriminator)

                # train generator fix discriminator
                # using a new random input
                # label size = batch_size
                generator_input_ = np.random.uniform(-1, 1, size = [FLAGS.batch_size,1,1, FLAGS.latent_dim])

                generator_feed_dict = {generator_input: generator_input_,
                                       is_training: True,
                                       keep_prob: 0.5}
                images_fake_ = sess.run(fetches=images_generated, feed_dict=generator_feed_dict)

                train_feed_dict_generator = {generator_input: generator_input_,
                                             is_training: True,
                                             keep_prob: 0.5,
                                             labels_real: labels_real_}

                _, generator_loss_value, generator_logits_value = \
                    sess.run([generator_train_op, generator_loss, generator_logits], feed_dict=train_feed_dict_generator)

                train_summary_str = sess.run(summary)

                duration = time.time() - start_time
                # if step%3==0:
                #     print('d_loss')
                #     print(discriminator_loss_value)
                #     print('gen_loss')
                #     print(generator_loss_value)

                if step%1000==0:
                    examples_per_sec = FLAGS.batch_size / duration
                    sec_per_batch = float(duration)

                    print('\n--- Normal training ---')
                    format_str = ('%s: step %d, discrim_loss= %.2f, adversarial_loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, discriminator_loss_value, generator_loss_value,
                                        examples_per_sec, sec_per_batch))
                    # print(generator_loss_logits)
                    # print(labels_real)
                    #
                    # save generated images
                    generator_feed_dict = {generator_input: z_fix,
                                           is_training: False,
                                           keep_prob: 0.5}
                    images_sample = sess.run(fetches=images_generated, feed_dict=generator_feed_dict)

                    output_filename = 'out_3_{0}.jpg'.format(step)
                    dcgan.imsave(images_sample, [8, FLAGS.batch_size//8], os.path.join(FLAGS.res_output_dir, output_filename))
                if step%10000==0:
                    saver.save(sess, os.path.join(FLAGS.model_ckpt_dir, "dcgan_mnist_3.ckpt"))
            train_writer.add_summary(train_summary_str, step)
            train_writer.flush()

            coord.request_stop()
            coord.join(threads)

def main(args):
    if FLAGS.testing:
        print("Test the model!!!")
        dcgan.test()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()