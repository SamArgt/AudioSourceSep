import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flow_models import utils
from flow_models import flow_builder
from pipeline import data_loader
import argparse
import os
import shutil
import datetime
import sys
import io
import librosa
import matplotlib.pyplot as plt
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def train(noise, mirrored_strategy, args, flow, optimizer, ds_dist, ds_val_dist,
          manager, manager_issues, train_summary_writer, test_summary_writer):
    # Custom Training Step
    # Adding the tf.function makes it about 10 times faster!!!

    with mirrored_strategy.scope():
        def compute_train_loss(X):
            X = X + tf.random.normal(X.shape) * noise
            per_example_loss = -flow.log_prob(X)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=args.batch_size)

        def compute_test_loss(X):
            X = X + tf.random.normal(X.shape) * noise
            per_example_loss = -flow.log_prob(X)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=args.test_batch_size)

    def train_step(inputs):
        with tf.GradientTape() as tape:
            tape.watch(flow.trainable_variables)
            loss = compute_train_loss(inputs)
        gradients = tape.gradient(loss, flow.trainable_variables)
        optimizer.apply_gradients(
            list(zip(gradients, flow.trainable_variables)))
        return loss

    def test_step(inputs):
        loss = compute_test_loss(inputs)
        return loss

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = mirrored_strategy.run(
            train_step, args=(dataset_inputs,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs):
        per_replica_losses = mirrored_strategy.run(
            test_step, args=(dataset_inputs,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def post_processing(x):
        if args.data_type == 'image':
            x = np.clip(x, 0., 255.)
            x = np.round(x, decimals=0).astype(int)
        else:
            x = np.clip(x, args.minval, args.maxval)
            if args.scale == "power":
                x = librosa.power_to_db(x)
        return x

    # Display first generated samples
    with mirrored_strategy.scope():
        samples = flow.sample(32)
    samples = post_processing(samples.numpy().reshape([32] + args.data_shape))
    try:
        figure = image_grid(samples, args.data_shape, args.data_type,
                            sampling_rate=args.sampling_rate, fmin=args.fmin, fmax=args.fmax)
        with train_summary_writer.as_default():
            tf.summary.image("32 generated samples", plot_to_image(figure), max_outputs=50, step=0)
    except IndexError:
        with train_summary_writer.as_default():
            tf.summary.text(name="display error",
                            data="Impossible to display spectrograms because of NaN values", step=0)

    N_EPOCHS = args.n_epochs
    batch_size = args.batch_size
    t0 = time.time()
    count_step = optimizer.iterations.numpy()
    min_val_loss = 1e6
    prev_history_loss_avg = None
    loss_per_epoch = 10  # number of losses per epoch to save
    is_nan_loss = False
    test_loss = tfk.metrics.Mean(name='test_loss')
    history_loss_avg = tf.keras.metrics.Mean(name="tensorboard_loss")
    epoch_loss_avg = tf.keras.metrics.Mean(name="epoch_loss")
    n_train = args.n_train
    print("Start Training on {} epochs".format(N_EPOCHS))
    # Custom Training Loop
    for epoch in range(1, N_EPOCHS + 1):
        epoch_loss_avg.reset_states()

        if is_nan_loss:
            break

        for batch in ds_dist:
            loss = distributed_train_step(batch)
            history_loss_avg.update_state(loss)
            epoch_loss_avg.update_state(loss)
            count_step += 1

            # every loss_per_epoch train step
            if count_step % (n_train // (batch_size * loss_per_epoch)) == 0:
                # check nan loss
                if (tf.math.is_nan(loss)) or (tf.math.is_inf(loss)):
                    print('Nan or Inf Loss: {}'.format(loss))
                    is_nan_loss = True
                    break

                # Save history and monitor it on tensorboard
                curr_loss_history = history_loss_avg.result()
                with train_summary_writer.as_default():
                    step_int = int(10 * count_step * batch_size / n_train)
                    tf.summary.scalar(
                        'loss', curr_loss_history, step=step_int)

                if manager_issues is not None:
                    # look for huge jump in the loss
                    if prev_history_loss_avg is None:
                        prev_history_loss_avg = curr_loss_history
                    elif curr_loss_history - prev_history_loss_avg > 10**6:
                        print("Huge gap in the loss")
                        save_path = manager_issues.save()
                        print("Model weights saved at {}".format(save_path))
                        with train_summary_writer.as_default():
                            tf.summary.text(name='Loss Jump',
                                            data=tf.constant(
                                                "Huge jump in the loss. Model weights saved at {}".format(save_path)),
                                            step=step_int)
                        prev_history_loss_avg = curr_loss_history

                history_loss_avg.reset_states()

        if (N_EPOCHS < 100) or (epoch % (N_EPOCHS // 100) == 0):
            # Compute validation loss and monitor it on tensorboard
            test_loss.reset_states()
            for elt in ds_val_dist:
                test_loss.update_state(distributed_test_step(elt))
            step_int = int(10 * count_step * batch_size / n_train)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=step_int)
            print("Epoch {:03d}: Train Loss: {:.3f} Val Loss: {:03f}".format(
                epoch, epoch_loss_avg.result(), test_loss.result()))
            # Generate some samples and visualize them on tensorboard
            with mirrored_strategy.scope():
                samples = flow.sample(32)
            samples = post_processing(samples.numpy().reshape([32] + args.data_shape))
            np.save(os.path.join("generated_samples", "generated_samples_{}".format(epoch)), samples)
            try:
                figure = image_grid(samples, args.data_shape, args.data_type,
                                    sampling_rate=args.sampling_rate, fmin=args.fmin, fmax=args.fmax)
                with train_summary_writer.as_default():
                    tf.summary.image("32 generated samples", plot_to_image(figure),
                                     max_outputs=50, step=epoch)
            except IndexError:
                with train_summary_writer.as_default():
                    tf.summary.text(name="display error",
                                    data="Impossible to display spectrograms because of NaN values",
                                    step=epoch)

            # If minimum validation loss is reached, save model
            curr_val_loss = test_loss.result()
            if curr_val_loss < min_val_loss:
                save_path = manager.save()
                print("Model Saved at {}".format(save_path))
                min_val_loss = curr_val_loss

    save_path = manager.save()
    print("Model Saved at {}".format(save_path))
    training_time = time.time() - t0
    return training_time, save_path


def setUp_optimizer(mirrored_strategy, args):
    lr = args.learning_rate
    with mirrored_strategy.scope():
        if args.optimizer == 'adam':
            optimizer = tfk.optimizers.Adam(
                lr=lr, clipvalue=args.clipvalue, clipnorm=args.clipnorm)
        elif args.optimizer == 'adamax':
            optimizer = tfk.optimizers.Adamax(lr=lr)
        else:
            raise ValueError("optimizer argument should be adam or adamax")
    return optimizer


def setUp_tensorboard():
    # Tensorboard
    # Clear any logs from previous runs
    try:
        shutil.rmtree('tensorboard_logs')
    except FileNotFoundError:
        pass
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(
        'tensorboard_logs', 'gradient_tape', current_time, 'train')
    test_log_dir = os.path.join(
        'tensorboard_logs', 'gradient_tape', current_time, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    return train_summary_writer, test_summary_writer


def setUp_checkpoint(mirrored_strategy, flow, optimizer):

    # Checkpoint object
    with mirrored_strategy.scope():
        ckpt = tf.train.Checkpoint(
            variables=flow.variables, optimizer=optimizer)
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=5)

    return ckpt, manager


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(sample, dataset):
    # Create a figure to contain the plot.
    f, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.flatten()
    if dataset == 'mnist':
        sample = sample[:, :, :, 0]
    for i, ax in enumerate(axes):
        ax.imshow(np.clip(sample[i] + 0.5, 0., 1.))
        ax.set_axis_off()
    return f


def main(args):

    # miscellaneous paramaters
    if args.dataset == 'mnist':
        args.data_shape = [32, 32, 1]
        args.data_type = "image"
    elif args.dataset == 'cifar10':
        args.data_shape = [32, 32, 3]
        args.data_type = "image"
    else:
        args.data_shape = [args.height, args.width, 1]
        args.dataset = os.path.abspath(args.dataset)
        args.data_type = "melspec"
        args.instrument = os.path.split(args.dataset)[-1]

    # Fine tune model serially from sigmaL to sigma1
    sigmas = np.logspace(np.log(args.sigmaL) / np.log(10), np.log(args.sigma1) / np.log(10), num=args.n_sigmas)
    abs_restore_path = os.path.abspath(args.RESTORE)

    if args.output == 'noise_conditioned_flows':
        output_dirname = 'glow_' + args.dataset + '_' + str(args.L) + '_' + \
            str(args.K) + '_' + str(args.n_filters) + \
            '_' + str(args.batch_size)
        output_dirpath = os.path.join(args.output, output_dirname)
    else:
        output_dirpath = args.output

    output_dirpath = os.path.abspath(output_dirpath)
    try:
        os.mkdir(output_dirpath)
        os.chdir(output_dirpath)
    except FileExistsError:
        os.chdir(output_dirpath)

    log_file = open('out.log', 'a')
    if args.debug is False:
        sys.stdout = log_file

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # Distributed Strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(
        mirrored_strategy.num_replicas_in_sync))

    # Load Dataset
    if args.data_type == "image":
        ds, ds_val, ds_dist, ds_val_dist, minibatch = data_loader.load_toydata(dataset=args.dataset, batch_size=args.batch_size,
                                                                               mirrored_strategy=mirrored_strategy)
        args.test_batch_size = 5000
        if args.dataset == "mnist":
            args.n_train = 60000
        else:
            args.n_train = 50000
        args.n_test = 10000

    else:
        ds, ds_val, ds_dist, ds_val_dist, minibatch, n_train, n_test = data_loader.load_melspec_ds(args.dataset + '/train',
                                                                                                   args.dataset + '/test',
                                                                                                   batch_size=args.batch_size,
                                                                                                   shuffle=True,
                                                                                                   mirrored_strategy=mirrored_strategy)
        args.test_batch_size = args.batch_size
        args.n_train = n_train
        args.n_test = n_test
        if args.scale == 'power':
            args.minval = 1e-10
            args.maxval = 100.
        elif args.scale == 'dB':
            args.minval = -100.
            args.maxval = 20.
        else:
            raise ValueError("scale should be 'power' or 'dB'")
        args.fmin = 125
        args.fmax = 7600
        args.sampling_rate = 16000

    # post processing
    def post_processing(x):
        if args.data_type == 'image':
            x = np.clip(x, 0., 255.)
            x = np.round(x, decimals=0).astype(int)
        else:
            x = np.clip(x, args.minval, args.maxval)
            if args.scale == "power":
                x = librosa.power_to_db(x)
        return x

    # Build Flow and Set up optimizer
    flow = flow_builder.build_glow(minibatch, args.data_shape, L=args.L, K=args.K, n_filters=args.n_filters,
                                   l2_reg=args.l2_reg, mirrored_strategy=mirrored_strategy, learntop=args.learntop,
                                   data_type=args.data_type, minval=args.minval, maxval=args.maxval,
                                   use_logit=args.use_logit, alpha=args.alpha)

    params_dict = vars(args)
    template = 'Glow Flow \n\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    with mirrored_strategy.scope():
        print("flow sample shape: ", flow.sample(1).shape)

    total_trainable_variables = utils.total_trainable_variables(flow)
    print("Total Trainable Variables: ", total_trainable_variables)

    # Set up optimizer
    optimizer = setUp_optimizer(mirrored_strategy, args)

    for sigma in sigmas:
        os.chdir(output_dirpath)
        try:
            os.mkdir('sigma_{}'.format(round(sigma, 2)))
            os.chdir('sigma_{}'.format(round(sigma, 2)))
        except FileExistsError:
            os.chdir('sigma_{}'.format(round(sigma, 2)))

        print("_" * 100)
        print("Training at noise level {}".format(round(sigma, 2)))
        # Set up tensorboard
        train_summary_writer, test_summary_writer = setUp_tensorboard()

        # Set up checkpoint
        ckpt, manager = setUp_checkpoint(
            mirrored_strategy, flow, optimizer)

        # restore
        with mirrored_strategy.scope():
            status = ckpt.restore(abs_restore_path)
            assert optimizer.iterations > 0
            status.assert_existing_objects_matched()
            print("Model Restored from {}".format(abs_restore_path))

        # Set up optimizer
        # optimizer = setUp_optimizer(mirrored_strategy, args)
        with train_summary_writer.as_default():
            sample = list(ds.take(1).as_numpy_iterator())[0]
            sample = sample[:32]
            figure = image_grid(sample, args.dataset)
            tf.summary.image("original images", plot_to_image(figure), max_outputs=1, step=0)

        params_dict = vars(args)
        template = 'Glow Flow \n\t '
        for k, v in params_dict.items():
            template += '{} = {} \n\t '.format(k, v)
        with train_summary_writer.as_default():
            tf.summary.text(name='Parameters',
                            data=tf.constant(template), step=0)
            tf.summary.text(name="Total Trainable Variables",
                            data=tf.constant(str(total_trainable_variables)), step=0)

        # Train
        training_time, save_path = train(sigma, mirrored_strategy, args, flow, optimizer, ds_dist, ds_val_dist,
                                         manager, None, train_summary_writer, test_summary_writer)
        print("Training time: ", np.round(training_time, 2), ' seconds')

        # Fine tune model serially
        abs_restore_path = os.path.abspath(save_path)
        args.RESTORE = abs_restore_path

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train noise conditioned Flow Model')

    parser.add_argument('RESTORE', type=str, default=None,
                        help='directory of saved weights (optional)')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10 or directory to melspec")
    parser.add_argument('--output', type=str, default='noise_conditioned_flows',
                        help='output dirpath for savings')
    parser.add_argument('--debug', action="store_true")

    # Noise parameters
    parser.add_argument('--sigma1', type=float, default=1.0)
    parser.add_argument('--sigmaL', type=float, default=0.01)
    parser.add_argument('--n_sigmas', type=int, default=10)

    # Model hyperparameters
    parser.add_argument('--L', default=3, type=int,
                        help='Depth level')
    parser.add_argument('--K', type=int, default=32,
                        help="Number of Step of Flow in each Block")
    parser.add_argument('--n_filters', type=int, default=512,
                        help="number of filters in the Convolutional Network")
    parser.add_argument('--l2_reg', type=float, default=None,
                        help="L2 regularization for the coupling layer")
    parser.add_argument("--learntop", action="store_true",
                        help="learnable prior distribution")

    # Optimization parameters
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipvalue', type=float, default=None,
                        help="Clip value for Adam optimizer")
    parser.add_argument('--clipnorm', type=float, default=None,
                        help='Clip norm for Adam optimize')

    # preprocessing parameters
    parser.add_argument('--use_logit', action="store_true",
                        help="Either to use logit function to preprocess the data")
    parser.add_argument('--alpha', type=float, default=10**(-6),
                        help='preprocessing parameter: x = logit(alpha + (1 - alpha) * z / 256.). Only if use logit')
    parser.add_argument('--noise', type=float, default=None,
                        help='noise level for BASIS separation')

    args = parser.parse_args()

    main(args)
