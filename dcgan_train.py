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

        # concat real images and generated images
        images_ = tf.concat([images_real, images_fake], axis=0)
        labels_ = tf.concat([labels_real, labels_fake], axis=0) + \
                  0.0*tf.truncated_normal(shape=[2*FLAGS.batch_size, 1,1,1])

        # shuffle images, labels pairs
        # size of images: 2* batch_size
        images, labels = [images_, labels_] # NEED to SHUFFLE!!!

        # BUILD GRAPH
        print("Add operators to graph")
        images_generated = dcgan.generator(generator_input, is_training, keep_prob, reuse=tf.AUTO_REUSE)
        # dcgan.generator(generator_input, is_training=is_training, keep_prob=keep_prob)
        _, discriminator_logits = dcgan.discriminator(images, is_training=is_training, keep_prob=keep_prob, reuse=False)
        _, generator_logits = dcgan.discriminator(images_generated, is_training=is_training, keep_prob=keep_prob, reuse=True)

        labels_real_ = np.ones([FLAGS.batch_size, 1, 1, 1])
        labels_fake_ = np.zeros([FLAGS.batch_size, 1, 1, 1])
        labels_debug = np.concatenate((labels_real_, labels_fake_), axis=0).astype('float32')

        discriminator_loss = dcgan.loss_cal(discriminator_logits, labels_debug, name="disc_loss")
        generator_loss = dcgan.loss_cal(generator_logits, labels_real_.astype('float32'), name="gen_loss")

        vars_trainable = tf.trainable_variables()
        vars_disctriminator = [var for var in vars_trainable if var.name.startswith('discriminator')]
        vars_generator = [var for var in vars_trainable if var.name.startswith('generator')]

        discriminator_train_op, global_step = dcgan.training(discriminator_loss, vars_disctriminator, learning_rate=0.00000005)
        generator_train_op, global_step = dcgan.training(generator_loss, vars_generator, learning_rate=0.0000002)

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


                generator_feed_dict = {generator_input: generator_input_,
                                       is_training: True,
                                       keep_prob: 0.5,
                                       reuse: True}
                # generate fake images
                images_fake_ = sess.run(fetches=images_generated, feed_dict=generator_feed_dict)

                # package the generated images with the feeded real images and shuffle
                images_batch, labels_batch = sess.run(fetches=[images, labels],
                                                      feed_dict={images_fake: images_fake_,
                                                                 labels_real: labels_real_,
                                                                 labels_fake: labels_fake_})

                train_feed_dict_discriminator = {images: images_batch,
                                                 labels: labels_batch,
                                                 is_training: True,
                                                 generator_input: generator_input_,
                                                 keep_prob: 0.5}

                start_time = time.time()

                _, discriminator_loss_value, discriminator_loss_logits= \
                    sess.run([discriminator_train_op, discriminator_loss, discriminator_logits], feed_dict=train_feed_dict_discriminator)

                # train generator fix discriminator
                # using a new random input
                # label size = batch_size
                generator_input_ = np.random.uniform(-1, 1, size = [FLAGS.batch_size,1,1, FLAGS.latent_dim])

                generator_feed_dict = {generator_input: generator_input_,
                                       is_training: True,
                                       keep_prob: 0.5}
                images_fake_ = sess.run(fetches=images_generated, feed_dict=generator_feed_dict)

                train_feed_dict_generator = {images_fake: images_fake_,
                                             generator_input: generator_input_,
                                             is_training: True,
                                             keep_prob: 0.5,
                                             labels_real: labels_real_,
                                             labels_fake: labels_fake_}

                _, generator_loss_value, generator_logits_value = \
                    sess.run([generator_train_op, generator_loss, generator_logits], feed_dict=train_feed_dict_generator)

                train_summary_str = sess.run(summary)

                duration = time.time() - start_time
                # if step%3==0:
                #     print('d_loss')
                #     print(discriminator_loss_value)
                #     print('gen_loss')
                #     print(generator_loss_value)

                if step%100==0:
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

                    output_filename = 'out_{0}.jpg'.format(step)
                    dcgan.imsave(images_sample, [10, FLAGS.batch_size//10], os.path.join(FLAGS.res_output_dir, output_filename))

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