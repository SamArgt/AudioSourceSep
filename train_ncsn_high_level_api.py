import numpy as np
import tensorflow as tf
from ncsn import score_network
from flow_models import utils
from pipeline import data_loader
import tensorflow_datasets as tfds
import argparse
import time
import datetime
import os
import sys
from train_utils import *
tfk = tf.keras


class CustomModel(tfk.Model):
    def __init__(self, args, name="ScoreNetworkModel"):
        super(CustomModel, self).__init__(name=name)
        self.scorenet = score_network.CondRefineNetDilated(args.data_shape, args.n_filters, args.num_classes, args.use_logit)
        self.data_shape = args.data_shape
        self.local_batch_size = args.local_batch_size

    def call(self, inputs, training=False):
        X, labels = inputs
        return self.scorenet(X, labels, training=training)

    def train_step(self, data):
        X, pertubed_X, labels, used_sigmas = data
        target = - (pertubed_X - X) / (used_sigmas ** 2)
        with tf.GradientTape() as tape:
            # tape.watch(self.trainable_variables)
            scores = self((pertubed_X, labels), training=True)
            loss = self.compiled_loss(target - scores, used_sigmas)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        X, pertubed_X, labels, used_sigmas = data
        target = - (pertubed_X - X) / (used_sigmas ** 2)
        scores = self((pertubed_X, labels), training=False)
        loss = self.compiled_loss(target - scores, used_sigmas)
        loss = tf.reduce_mean(loss)
        return {'loss': loss}

    def sample(self, n_samples, sigmas, n_steps_each=100, step_lr=0.00002, return_arr=False):
        """
        Anneal Langevin dynamics
        """
        x_mod = tf.random.uniform([n_samples] + list(self.data_shape))
        if return_arr:
            x_arr = [x_mod]
        for i, sigma in enumerate(sigmas):
            labels = tf.expand_dims(tf.ones(n_samples) * i, -1)
            step_size = tf.constant(step_lr * (sigma / sigmas[-1]) ** 2, dtype=tf.float32)
            for s in range(n_steps_each):
                noise = tf.random.normal((n_samples,)) * tf.math.sqrt(step_size * 2)
                grad = self((x_mod, labels))
                x_mod = x_mod + step_size * grad + tf.reshape(noise, (n_samples, 1, 1, 1))
                if return_arr:
                    x_arr.append(tf.clip_by_value(x_mod, 0., 1.))

        if return_arr:
            return x_arr
        else:
            return tf.clip_by_value(x_mod, 0., 1.)


class CustomLoss(tfk.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, target_scores_diff, used_sigmas):
        return tf.reduce_sum(tf.square(target_scores_diff) / 2., axis=[1, 2, 3]) * (used_sigmas ** 2.)


def main(args):

    sigmas_np = np.logspace(np.log(args.sigma1) / np.log(10),
                            np.log(args.sigmaL) / np.log(10),
                            num=args.num_classes)

    sigmas_tf = tf.constant(sigmas_np, dtype=tf.float32)

    # miscellaneous paramaters
    if args.dataset == 'mnist':
        args.data_shape = [32, 32, 1]
        args.img_type = "image"
        args.preprocessing_glow = None
    elif args.dataset == 'cifar10':
        args.data_shape = [32, 32, 3]
        args.img_type = "image"
        args.preprocessing_glow = None
    else:
        args.data_shape = [args.height, args.width, 1]
        args.dataset = os.path.abspath(args.dataset)
        args.img_type = "melspec"
        args.preprocessing_glow = "melspec"
        args.instrument = os.path.split(args.dataset)[-1]

    # output directory
    if args.restore is not None:
        abs_restore_path = os.path.abspath(args.restore)
    if args.output == 'trained_ncsn':
        if args.dataset != 'mnist' and args.dataset != 'cifar10':
            dataset = args.instrument
        else:
            dataset = args.dataset

        output_dirname = 'ncsn' + '_' + dataset + '_' + \
            str(args.n_filters) + '_' + str(args.batch_size)

        if args.use_logit:
            output_dirname += '_logit'
        if args.restore is not None:
            output_dirname += '_ctd'

        output_dirpath = os.path.join(args.output, output_dirname)
    else:
        output_dirpath = args.output
    try:
        os.mkdir(output_dirpath)
        os.chdir(output_dirpath)
    except FileExistsError:
        os.chdir(output_dirpath)
    log_file = open('out.log', 'w')
    if args.debug is False:
        sys.stdout = log_file

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # Distributed Strategy
    if args.mirrored_strategy:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(
            mirrored_strategy.num_replicas_in_sync))
        args.local_batch_size = args.batch_size // mirrored_strategy.num_replicas_in_sync
    else:
        mirrored_strategy = None
        args.local_batch_size = args.batch_size

    # Load Dataset
    if (args.dataset == "mnist") or (args.dataset == "cifar10"):
        datasets, info = tfds.load(
            name=args.dataset, with_info=True, as_supervised=False)
        ds_train, ds_test = datasets['train'], datasets['test']
        ds_train = ds_train.map(lambda x: x['image'])
        ds_test = ds_test.map(lambda x: x['image'])
        BUFFER_SIZE = 10000
        BATCH_SIZE = args.batch_size

        def preprocess(image):
            sigma_idx = tf.random.uniform(shape=(), maxval=10, dtype=tf.int32)
            used_sigma = tf.gather(params=sigmas_tf, indices=sigma_idx)
            X = tf.cast(image, tf.float32)
            X = tf.pad(X, tf.constant([[2, 2], [2, 2], [0, 0]]))
            X = X / 256. + tf.random.uniform(X.shape) / 256.
            pertubed_X = X + tf.random.normal(X.shape) * used_sigma
            inputs = {'pertubed_X': pertubed_X, 'sigma_idx': sigma_idx}
            target = -(pertubed_X - X) / (used_sigma ** 2)
            sample_weight = used_sigma ** 2
            return inputs, target, sample_weight

        train_dataset = ds_train.map(preprocess).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        eval_dataset = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    else:
        ds, ds_val, ds_dist, ds_val_dist, minibatch, n_train = data_loader.load_melspec_ds(args.dataset, batch_size=args.batch_size,
                                                                                           reshuffle=True, model='ncsn',
                                                                                           num_classes=args.num_classes,
                                                                                           mirrored_strategy=mirrored_strategy,
                                                                                           use_logt=args.use_logit)
    # Display original images
    # Clear any logs from previous runs
    try:
        shutil.rmtree('logs')
    except FileNotFoundError:
        pass
    logdir = os.path.join("logs", "samples") + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
        inputs, _, _ = list(train_dataset.take(1).as_numpy_iterator())[0]
        sample = inputs["pertubed_X"]
        sample = sample[:32]
        figure = image_grid(sample, args.data_shape, args.img_type,
                            sampling_rate=args.sampling_rate, fmin=args.fmin, fmax=args.fmax)
        tf.summary.image("pertubed images", plot_to_image(
            figure), max_outputs=1, step=0)

    # Set up optimizer
    optimizer = setUp_optimizer(mirrored_strategy, args)

    # Build ScoreNet
    """
    if mirrored_strategy is not None:
        with mirrored_strategy.scope():
            model = CustomModel(args)
    else:
        model = CustomModel(args)
    """
    pertubed_X = tfk.Input(shape=args.data_shape, dtype=tf.float32, name="pertubed_X")
    sigma_idx = tfk.Input(shape=None, dtype=tf.int32, name="sigma_idx")
    outputs = score_network.CondRefineNetDilated(args.data_shape, args.n_filters,
                                                 args.num_classes, args.use_logit)(pertubed_X, sigma_idx)
    model = tfk.Model(inputs=[pertubed_X, sigma_idx], outputs=outputs, name="ScoreNetwork")
    model.compile(optimizer=optimizer, loss=tfk.losses.MeanSquaredError())
    # model.build(([None] + list(args.data_shape), [None]))
    # print(model.summary())

    # Set up callbacks
    logdir = os.path.join("logs", "scalars") + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=logdir, write_graph=True, update_freq="epoch",
                                                     profile_batch=2, embeddings_freq=0, histogram_freq=0)

    def display_generated_samples(epoch, logs):
        if (args.n_epochs < 10) or ((epoch + 1) % (args.n_epochs // 10) == 0):
            if mirrored_strategy is not None:
                with mirrored_strategy.scope():
                    gen_samples = model.sample(32, sigmas_np)
            else:
                gen_samples = model.sample(32, sigmas_np)

            figure = image_grid(gen_samples, args.data_shape, args.img_type,
                                sampling_rate=args.sampling_rate, fmin=args.fmin, fmax=args.fmax)
            sample_image = plot_to_image(figure)
            with file_writer.as_default():
                tf.summary.image("Generated Samples", sample_image, step=epoch)

        else:
            pass

    gen_samples_callback = tfk.callbacks.LambdaCallback(on_epoch_end=display_generated_samples)

    callbacks = [
        tensorboard_callback,
        tfk.callbacks.ModelCheckpoint(filepath="tf_ckpts/",
                                      save_weights_only=True,
                                      monitor="val_loss",
                                      save_best_only=True),
        tfk.callbacks.TerminateOnNaN(),
        tfk.callbacks.EarlyStopping(monitor='val_loss', min_delta=10, patience=10),
        gen_samples_callback
    ]
    # restore

    # Display parameters
    params_dict = vars(args)
    template = ''
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    with file_writer.as_default():
        tf.summary.text(name='Parameters',
                        data=tf.constant(template), step=0)

    # Train
    t0 = time.time()
    model.fit(train_dataset, epochs=args.n_epochs, batch_size=args.batch_size,
              validation_data=eval_dataset, callbacks=callbacks)

    total_trainable_variables = utils.total_trainable_variables(model)
    print("Total Trainable Variables: ", total_trainable_variables)

    print("Training time: ", np.round(time.time() - t0, 2), ' seconds')

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Flow model')
    # dataset parameters
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10 or directory to tfrecords")

    parser.add_argument('--mirrored_strategy', action="store_false")

    # Spectrograms Parameters
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--fmin", type=int, default=125)
    parser.add_argument("--fmax", type=int, default=7600)

    # Output and Restore Directory
    parser.add_argument('--output', type=str, default='trained_ncsn',
                        help='output dirpath for savings')
    parser.add_argument('--restore', type=str, default=None,
                        help='directory of saved weights (optional)')
    parser.add_argument('--latest', action="store_true", help="Restore latest checkpoint from restore directory")
    parser.add_argument('--debug', action="store_true")

    # Model hyperparameters
    parser.add_argument("--n_filters", type=int, default=64,
                        help='number of filters in the Score Network')
    parser.add_argument("--sigma1", type=float, default=1.)
    parser.add_argument("--sigmaL", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=10)

    # Optimization parameters
    parser.add_argument('--n_epochs', type=int, default=400,
                        help='number of epochs to train')
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="adam or adamax")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipvalue', type=float, default=None,
                        help="Clip value for Adam optimizer")
    parser.add_argument('--clipnorm', type=float, default=None,
                        help='Clip norm for Adam optimize')

    # Preprocessing parameters
    parser.add_argument("--use_logit", action="store_true")

    args = parser.parse_args()

    main(args)
