import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from flow_models import flow_tfk_layers
from flow_models import flow_glow
from flow_models import flow_real_nvp
from flow_models import flow_tfp_bijectors
from flow_models import utils
import argparse
import time
import os
import sys
import shutil
import datetime
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def load_data(mirrored_strategy, args):

    if args.dataset == 'mnist':
        data_shape = (28, 28, 1)
    elif args.dataset == 'cifar10':
        data_shape = (32, 32, 3)
    else:
        raise ValueError("args.dataset should be mnist or cifar10")
    buffer_size = 2048
    global_batch_size = args.batch_size
    ds = tfds.load(args.dataset, split='train', shuffle_files=True)
    # Build your input pipeline
    ds = ds.map(lambda x: x['image'])
    ds = ds.map(lambda x: tf.cast(x, tf.float32))
    if args.use_logit:
        ds = ds.map(lambda x: args.alpha + (1 - args.alpha) * x / 256.)
        ds = ds.map(lambda x: x + tf.random.uniform(shape=data_shape,
                                                    minval=0., maxval=1. / 256.))
        ds = ds.map(lambda x: tf.math.log(x / (1 - x)))
    else:
        ds = ds.map(lambda x: x / 256. - 0.5)
        ds = ds.map(lambda x: x + tf.random.uniform(shape=data_shape,
                                                    minval=0., maxval=1. / 256.))

    if args.noise is not None:
        ds = ds.map(lambda x: x + tf.random.normal(shape=data_shape, mean=0, stddev=args.noise))

    ds = ds.shuffle(buffer_size).batch(global_batch_size, drop_remainder=True)
    minibatch = list(ds.take(1).as_numpy_iterator())[0]
    ds_dist = mirrored_strategy.experimental_distribute_dataset(ds)
    # Validation Set
    ds_val = tfds.load(args.dataset, split='test', shuffle_files=True)
    ds_val = ds_val.map(lambda x: x['image'])
    ds_val = ds_val.map(lambda x: tf.cast(x, tf.float32))
    if args.use_logit:
        ds_val = ds_val.map(lambda x: args.alpha + (1 - args.alpha) * x / 256.)
        ds_val = ds_val.map(lambda x: x + tf.random.uniform(shape=data_shape, minval=0., maxval=1. / 256.))
        ds_val = ds_val.map(lambda x: tf.math.log(x / (1 - x)))
    else:
        ds_val = ds_val.map(lambda x: x / 256. - 0.5)
        ds_val = ds_val.map(lambda x: x + tf.random.uniform(shape=data_shape, minval=0., maxval=1. / 256.))

    if args.noise is not None:
        ds_val = ds_val.map(lambda x: x + tf.random.normal(shape=data_shape, mean=0, stddev=args.noise))
    ds_val = ds_val.batch(5000)
    ds_val_dist = mirrored_strategy.experimental_distribute_dataset(ds_val)

    return ds_dist, ds_val_dist, minibatch


def build_flow(mirrored_strategy, args, minibatch):
    tfk.backend.clear_session()

    # Set flow parameters
    if args.dataset == 'mnist':
        data_shape = [28, 28, 1]
    elif args.dataset == 'cifar10':
        data_shape = [32, 32, 3]
    if args.L == 2:
        base_distr_shape = [data_shape[0] // 4, data_shape[1] // 4, data_shape[2] * 16]
    elif args.L == 3:
        base_distr_shape = [data_shape[0] // 8, data_shape[1] // 8, data_shape[2] * 32]
    else:
        raise ValueError("L should be 2 or 3")

    shift_and_log_scale_layer = flow_tfk_layers.ShiftAndLogScaleResNet

    # Build Flow and Optimizer
    with mirrored_strategy.scope():
        if args.L == 2:
            bijector = flow_glow.GlowBijector_2blocks(args.K, data_shape,
                                                      shift_and_log_scale_layer,
                                                      args.n_filters, minibatch, **{'l2_reg': args.l2_reg})
        elif args.L == 3:
            bijector = flow_glow.GlowBijector_3blocks(args.K, data_shape,
                                                      shift_and_log_scale_layer,
                                                      args.n_filters, minibatch, **{'l2_reg': args.l2_reg})
        inv_bijector = tfb.Invert(bijector)
        flow = tfd.TransformedDistribution(tfd.Normal(
            0., 1.), inv_bijector, event_shape=base_distr_shape)

    return flow


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
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=5000)

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
    if args.dataset == 'mnist':
        n_train = 60000
    elif args.dataset == 'cifar10':
        n_train = 50000
    print("Start Training on {} epochs".format(N_EPOCHS))
    # Custom Training Loop
    for epoch in range(N_EPOCHS):
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
            # Generate some samples and visualize them on tensoboard
            with mirrored_strategy.scope():
                samples = flow.sample(9)
            samples = samples.numpy().reshape((9, 28, 28, 1))
            with train_summary_writer.as_default():
                tf.summary.image("9 generated samples", samples,
                                 max_outputs=27, step=epoch)
            # If minimum validation loss is reached, save model
            curr_val_loss = test_loss.result()
            if curr_val_loss < min_val_loss:
                save_path = manager.save()
                print("Model Saved at {}".format(save_path))
                min_val_loss = curr_val_loss

    training_time = time.time() - t0
    return training_time


def main(args):

    if args.output == 'mnist_trained_flow':
        if args.restore is None:
            output_dirname = 'glow_' + args.dataset + '_' + str(args.L) + '_' + \
                str(args.K) + '_' + str(args.n_filters) + '_' + str(args.batch_size)
            if args.use_logit:
                output_dirname += '_logit'
        else:
            _, output_dirname = os.path.split(args.restore)
            output_dirname += '_ctd'
            abs_restore_path = os.path.join(os.path.abspath(args.restore), 'tf_ckpts')

        output_dirpath = os.path.join(args.output, output_dirname)
    else:
        output_dirpath = args.output

    try:
        os.mkdir(output_dirpath)
        os.chdir(output_dirpath)
    except FileExistsError:
        os.chdir(output_dirpath)

    log_file = open('out.log', 'w')
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
    ds_dist, ds_val_dist, minibatch = load_data(mirrored_strategy, args)

    # Build Flow and Set up optimizer
    flow = build_flow(mirrored_strategy, args, minibatch)

    # Set up optimizer
    optimizer = setUp_optimizer(mirrored_strategy, args)

    # Set up checkpoint
    ckpt, manager, manager_issues = setUp_checkpoint(
        mirrored_strategy, args, flow, optimizer)

    # restore
    if args.restore is not None:
        checkpoint_path = tf.train.latest_checkpoint(abs_restore_path)
        with mirrored_strategy.scope():
            ckpt.restore(checkpoint_path)
            print("Model Restored")

    params_dict = vars(args)
    template = 'Glow Flow \n\t '
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
    training_time = train(mirrored_strategy, args, flow, optimizer, ds_dist, ds_val_dist,
                          manager, manager_issues, train_summary_writer, test_summary_writer)
    print("Training time: ", np.round(training_time, 2), ' seconds')

    # Saving the last variables
    save_path = manager.save()
    print("Model Saved at {}".format(save_path))

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Flow model on MNIST dataset')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10")
    parser.add_argument('--output', type=str, default='mnist_trained_flow',
                        help='output dirpath for savings')
    parser.add_argument('--restore', type=str, default=None,
                        help='directory of saved weights (optional)')

    # Model hyperparameters
    parser.add_argument('--L', default=2, type=int,
                        help='Depth level')
    parser.add_argument('--K', type=int, default=16,
                        help="Number of Step of Flow in each Block")
    parser.add_argument('--n_filters', type=int, default=256,
                        help="number of filters in the Convolutional Network")
    parser.add_argument('--l2_reg', type=float, default=None,
                        help="L2 regularization for the coupling layer")

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