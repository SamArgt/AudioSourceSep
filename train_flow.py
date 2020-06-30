
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flow_models import utils
from flow_models import flow_builder
from pipeline import data_loader
import argparse
import time
import os
import sys
import shutil
import datetime
import matplotlib.pyplot as plt
import io
import librosa
from librosa.display import specshow
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def setUp_optimizer(mirrored_strategy, args):
    lr = args.learning_rate
    with mirrored_strategy.scope():
        if args.optimizer == 'adam':
            optimizer = tfk.optimizers.Adam(lr=lr, clipvalue=args.clipvalue, clipnorm=args.clipnorm)
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


def setUp_checkpoint(mirrored_strategy, args, flow, optimizer):

    # Checkpoint object
    with mirrored_strategy.scope():
        ckpt = tf.train.Checkpoint(
            variables=flow.variables, optimizer=optimizer)
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=5)
        # Debugging: if huge jump in the loss, save weights here
        manager_issues = tf.train.CheckpointManager(
            ckpt, './tf_ckpts_issues', max_to_keep=3)

    return ckpt, manager, manager_issues


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


def image_grid(sample, data_shape, img_type="image", **kwargs):
    # Create a figure to contain the plot.
    f, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.flatten()
    if data_shape[-1] == 1:
        sample = sample[:, :, :, 0]
    for i, ax in enumerate(axes):
        if img_type == "image":
            ax.imshow(np.clip(sample[i] + 0.5, 0., 1.))
            ax.set_axis_off()
        elif img_type == "melspec":
            sample[i] = np.exp(sample[i])
            spec_db_sample = librosa.power_to_db(sample[i])
            specshow(spec_db_sample, sr=44100, ax=ax, x_axis='off', y_axis='off')

    return f


def train(mirrored_strategy, args, flow, optimizer, ds_dist, ds_val_dist,
          manager, manager_issues, train_summary_writer, test_summary_writer):
    # Custom Training Step
    # Adding the tf.function makes it about 10 times faster!!!

    with mirrored_strategy.scope():
        def compute_train_loss(X):
            per_example_loss = -flow.log_prob(X)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=args.batch_size)

        def compute_test_loss(X):
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

    with mirrored_strategy.scope():
        samples = flow.sample(32)
    samples = samples.numpy().reshape([32] + args.data_shape)
    try:
        figure = image_grid(samples, args.data_shape, args.img_type)
        with train_summary_writer.as_default():
            tf.summary.image("32 generated samples", plot_to_image(figure), max_outputs=50, step=0)
    except IndexError:
        with train_summary_writer.as_default():
            tf.summary.text(name="display error",
                            data="Impossible to display spectrograms because of NaN values", step=0)

    N_EPOCHS = args.n_epochs
    batch_size = args.batch_size
    t0 = time.time()
    loss_history = []
    count_step = optimizer.iterations.numpy()
    min_val_loss = 0.
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
                if tf.math.is_nan(loss):
                    print('Nan Loss')
                    is_nan_loss = True
                    break

                # Save history and monitor it on tensorboard
                curr_loss_history = history_loss_avg.result()
                loss_history.append(curr_loss_history)
                with train_summary_writer.as_default():
                    step_int = int(loss_per_epoch * count_step * batch_size / n_train)
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

        # every 10 epochs
        if (N_EPOCHS < 100) or (epoch % (N_EPOCHS // 100) == 0):
            # Compute validation loss and monitor it on tensoboard
            test_loss.reset_states()
            for elt in ds_val_dist:
                test_loss.update_state(distributed_test_step(elt))
            step_int = int(loss_per_epoch * count_step * batch_size / n_train)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=step_int)
            print("Epoch {:03d}: Train Loss: {:.3f} Val Loss: {:03f}".format(
                epoch, epoch_loss_avg.result(), test_loss.result()))
            # Generate some samples and visualize them on tensorboard
            with mirrored_strategy.scope():
                samples = flow.sample(32)
            samples = samples.numpy().reshape([32] + args.data_shape)
            try:
                figure = image_grid(samples, args.data_shape, args.img_type)
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


def main(args):
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

    # output directory
    if args.restore is not None:
        abs_restore_path = os.path.abspath(args.restore)

    if args.output == 'trained_flow':
        if args.dataset != 'mnist' and args.dataset != 'cifar10':
            dataset = args.instrument
        else:
            dataset = args.dataset
        if args.model == 'glow':
            output_dirname = args.model + '_' + dataset + '_' + str(args.L) + '_' + \
                str(args.K) + '_' + str(args.n_filters) + '_' + str(args.batch_size)
        elif args.model == 'flowpp':
            output_dirname = args.model + '_' + dataset + '_' + str(args.n_components) + '_' + \
                str(args.n_blocks_flow) + '_' + str(args.filters) + '_' + str(args.batch_size)

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

    # Set up tensorboard
    train_summary_writer, test_summary_writer = setUp_tensorboard()

    # Distributed Strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(
        mirrored_strategy.num_replicas_in_sync))

    # Load Dataset
    if args.model == 'flowpp':
        preprocessing_dataloader = False
    else:
        preprocessing_dataloader = True

    if (args.dataset == "mnist") or (args.dataset == "cifar10"):
        ds, ds_val, ds_dist, ds_val_dist, minibatch, n_train = data_loader.load_data(dataset=args.dataset, batch_size=args.batch_size,
                                                                                     mirrored_strategy=mirrored_strategy, use_logit=args.use_logit,
                                                                                     noise=args.noise, alpha=args.alpha, preprocessing=preprocessing_dataloader)
        args.test_batch_size = 5000
    else:
        ds, ds_val, ds_dist, ds_val_dist, minibatch, n_train = data_loader.load_melspec_ds(args.dataset, batch_size=args.batch_size,
                                                                                           reshuffle=True,
                                                                                           mirrored_strategy=mirrored_strategy)
        args.test_batch_size = args.batch_size
    args.n_train = n_train

    with train_summary_writer.as_default():
        sample = list(ds.take(1).as_numpy_iterator())[0]
        sample = sample[:32]
        figure = image_grid(sample, args.data_shape, args.img_type)
        tf.summary.image("original images", plot_to_image(figure), max_outputs=1, step=0)

    # Build Flow
    if args.model == 'glow':
        flow = flow_builder.build_glow(minibatch, args.data_shape, L=args.L, K=args.K, n_filters=args.n_filters,
                                       l2_reg=args.l2_reg, mirrored_strategy=mirrored_strategy, learntop=args.learntop,
                                       preprocessing=args.preprocessing_glow)
    elif args.model == 'flowpp':
        flow = flow_builder.build_flowpp(minibatch, args.data_shape, n_components=args.n_components,
                                         n_blocks_flow=args.n_blocks_flow, n_blocks_dequant=args.n_blocks_dequant,
                                         filters=args.filters, dropout_p=args.dropout_p, heads=args.heads,
                                         mirrored_strategy=mirrored_strategy)
    else:
        raise ValueError("model should be glow or flowpp")

    # Set up optimizer
    optimizer = setUp_optimizer(mirrored_strategy, args)

    # Set up checkpoint
    ckpt, manager, manager_issues = setUp_checkpoint(
        mirrored_strategy, args, flow, optimizer)

    # restore
    if args.restore is not None:
        with mirrored_strategy.scope():
            if args.latest:
                checkpoint_restore_path = tf.train.latest_checkpoint(abs_restore_path)
                assert checkpoint_restore_path is not None, abs_restore_path
            else:
                checkpoint_restore_path = abs_restore_path
            status = ckpt.restore(checkpoint_restore_path)
            status.assert_existing_objects_matched()
            assert optimizer.iterations > 0
            print("Model Restored from {}".format(abs_restore_path))

    params_dict = vars(args)
    template = ''
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)

    with mirrored_strategy.scope():
        print("flow sample shape: ", flow.sample(1).shape)

    total_trainable_variables = utils.total_trainable_variables(flow)
    print("Total Trainable Variables: ", total_trainable_variables)

    with train_summary_writer.as_default():
        tf.summary.text(name='Parameters',
                        data=tf.constant(template), step=0)
        tf.summary.text(name="Total Trainable Variables",
                        data=tf.constant(str(total_trainable_variables)), step=0)

    # Train
    training_time, _ = train(mirrored_strategy, args, flow, optimizer, ds_dist, ds_val_dist,
                             manager, manager_issues, train_summary_writer, test_summary_writer)
    print("Training time: ", np.round(training_time, 2), ' seconds')

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Flow model')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10 or directory to tfrecords")

    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--instrument", type=str, default="piano")

    parser.add_argument('--output', type=str, default='trained_flow',
                        help='output dirpath for savings')
    parser.add_argument('--restore', type=str, default=None,
                        help='directory of saved weights (optional)')
    parser.add_argument('--latest', action="store_true", help="Restore latest checkpoint from restore directory")
    parser.add_argument('--debug', action="store_true")

    # Model hyperparameters
    parser.add_argument('--model', default='glow', type=str, help='glow or flowpp')
    parser.add_argument("--learntop", action="store_true",
                        help="learnable prior distribution")
    # Glow hyperparameters
    parser.add_argument('--L', default=3, type=int,
                        help='Depth level')
    parser.add_argument('--K', type=int, default=32,
                        help="Number of Step of Flow in each Block")
    parser.add_argument('--n_filters', type=int, default=256,
                        help="number of filters in the Convolutional Network")
    parser.add_argument('--l2_reg', type=float, default=None,
                        help="L2 regularization for the coupling layer")
    # Flow++ hyperparameters
    parser.add_argument('--n_components', type=int, default=32)
    parser.add_argument('--n_blocks_flow', type=int, default=10)
    parser.add_argument('--n_blocks_dequant', type=int, default=2)
    parser.add_argument('--filters', type=int, default=96)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=4)

    # Optimization parameters
    parser.add_argument('--n_epochs', type=int, default=100,
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
    parser.add_argument('--noise', type=float, default=None, help='noise level for BASIS separation')

    args = parser.parse_args()

    main(args)
