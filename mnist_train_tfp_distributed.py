import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from flow_models import flow_tfk_layers
from flow_models import flow_glow
from flow_models import flow_real_nvp
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

def main():

    parser = argparse.ArgumentParser(
        description='Train Flow model on MNIST dataset')
    parser.add_argument('OUTPUT', type=str,
                        help='output dirpath for savings')
    parser.add_argument('--n_epochs', type=str, default=None,
                        help='number of epochs to train')
    parser.add_argument('--restore', type=str, default=None,
                        help='directory of saved weights (optional)')
    args = parser.parse_args()
    output_dirpath = args.OUTPUT
    if args.restore is not None:
        restore_abs_dirpath = os.path.abspath(args.restore)

    try:
        os.mkdir(output_dirpath)
        os.chdir(output_dirpath)
    except FileExistsError:
        os.chdir(output_dirpath)

    log_file = open('out.log', 'w')
    # sys.stdout = log_file

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

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

    tfk.backend.clear_session()

    # Distributed Strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(
        mirrored_strategy.num_replicas_in_sync))

    # Construct a tf.data.Dataset
    buffer_size = 2048
    batch_size_per_replica = 64
    global_batch_size = batch_size_per_replica * \
        mirrored_strategy.num_replicas_in_sync
    ds = tfds.load('mnist', split='train', shuffle_files=True)
    # Build your input pipeline
    ds = ds.map(lambda x: x['image'])
    ds = ds.map(lambda x: tf.cast(x, tf.float32))
    ds = ds.map(lambda x: x + tf.random.uniform(shape=(28, 28, 1),
                                                minval=0., maxval=1. / 256.))
    ds = ds.map(lambda x: x / 256.)
    ds = ds.shuffle(buffer_size).batch(global_batch_size)
    minibatch = list(ds.take(1).as_numpy_iterator())[0]
    ds_dist = mirrored_strategy.experimental_distribute_dataset(ds)
    # Validation Set
    ds_val = tfds.load('mnist', split='test', shuffle_files=True)
    ds_val = ds_val.map(lambda x: x['image'])
    ds_val = ds_val.map(lambda x: tf.cast(x, tf.float32))
    ds_val = ds_val.map(
        lambda x: x + tf.random.uniform(shape=(28, 28, 1), minval=0., maxval=1. / 256.))
    ds_val = ds_val.map(lambda x: x / 256.)
    ds_val = ds_val.batch(5000)
    ds_val_dist = mirrored_strategy.experimental_distribute_dataset(ds_val)

    # Set flow parameters
    data_shape = [28, 28, 1]  # (H, W, C)
    base_distr_shape = (7, 7, 16)  # (H//4, W//4, C*16)
    K = 8
    shift_and_log_scale_layer = flow_tfk_layers.ShiftAndLogScaleResNet
    n_filters_base = 32

    # Build Flow
    with mirrored_strategy.scope():
        # prepocessing_bijector = flow_tfp_bijectors.Preprocessing(data_shape)
        # minibatch = prepocessing_bijector.forward(tf.reshape(minibatch, (1, 28, 28, 1)))
        # minibatch = tf.reshape(minibatch, (28, 28, 1))
        flow_bijector = flow_glow.GlowBijector_2blocks(K, data_shape,
                                                       shift_and_log_scale_layer, n_filters_base, minibatch)
        # bijector = tfb.Chain([flow_bijector, prepocessing_bijector])
        inv_bijector = tfb.Invert(flow_bijector)
        flow = tfd.TransformedDistribution(tfd.Normal(
            0., 1.), inv_bijector, event_shape=base_distr_shape)

    params_str = 'Glow Bijector 2 Blocks: \n\t K = {} \n\t ShiftAndLogScaleResNet \n\t n_filters = {} \n\t batch size = {}'.format(
        K, n_filters_base, global_batch_size)
    print(params_str)
    with train_summary_writer.as_default():
        tf.summary.text(name='Flow parameters',
                        data=tf.constant(params_str), step=0)
    with mirrored_strategy.scope():
        print("flow sample shape: ", flow.sample(1).shape)
        # utils.print_summary(flow)
        print("Total Trainable Variables: ",
              utils.total_trainable_variables(flow))

    # Custom Training Step
    # Adding the tf.function makes it about 10 times faster!!!
    with mirrored_strategy.scope():
        def compute_loss(X):
            per_example_loss = -flow.log_prob(X)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

    with mirrored_strategy.scope():
        test_loss = tfk.metrics.Mean(name='test_loss')
        history_loss_avg = tf.keras.metrics.Mean(name="tensorboard_loss")
        epoch_loss_avg = tf.keras.metrics.Mean(nam="epoch_loss")

    def train_step(inputs):
        with tf.GradientTape() as tape:
            tape.watch(flow.trainable_variables)
            loss = compute_loss(inputs)
        gradients = tape.gradient(loss, flow.trainable_variables)
        optimizer.apply_gradients(
            list(zip(gradients, flow.trainable_variables)))
        history_loss_avg.update_state(loss)
        return loss

    def test_step(inputs):
        loss = compute_loss(inputs)
        test_loss.update_state(loss)

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = mirrored_strategy.run(
            train_step, args=(dataset_inputs,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs):
        return mirrored_strategy.run(test_step, args=(dataset_inputs,))

    # Training Parameters
    N_EPOCHS = 100
    if args.n_epochs is not None:
        N_EPOCHS = int(args.n_epochs)
    with mirrored_strategy.scope():
        optimizer = tf.keras.optimizers.Adam()

    print("Start Training on {} epochs".format(N_EPOCHS))

    # Checkpoint object
    with mirrored_strategy.scope():
        ckpt = tf.train.Checkpoint(
            variables=flow.variables, optimizer=optimizer)
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=5)
        # if huge jump in the loss, save weights here
        manage_issues = tf.train.CheckpointManager(
            ckpt, './tf_ckpts_issues', max_to_keep=3)
        # Restore weights if specified
        if args.restore is not None:
            ckpt.restore(tf.train.latest_checkpoint(
                os.path.join(restore_abs_dirpath, 'tf_ckpts')))

    t0 = time.time()
    loss_history = []
    count_step = optimizer.iterations.numpy()
    min_val_loss = 0.
    prev_history_loss_avg = None
    loss_per_epoch = 10  # number of losses per epoch to save
    is_nan_loss = False
    # Custom Training Loop
    for epoch in range(N_EPOCHS):
        epoch_loss_avg.reset_states()

        if is_nan_loss:
            break

        for batch in ds_dist:
            loss = distributed_train_step(batch)
            epoch_loss_avg.update_state(loss)
            count_step += 1

            # every loss_per_epoch train step
            if count_step % (60000 // (global_batch_size * loss_per_epoch)) == 0:
                # check nan loss
                if tf.math.is_nan(loss):
                    print('Nan Loss')
                    is_nan_loss = True
                    break

                # Save history and monitor it on tensorboard
                curr_loss_history = history_loss_avg.result()
                loss_history.append(curr_loss_history)
                with train_summary_writer.as_default():
                    step_int = int(loss_per_epoch * count_step * global_batch_size / 60000)
                    tf.summary.scalar(
                        'loss', curr_loss_history, step=step_int)

                # look for huge jump in the loss
                if prev_history_loss_avg is None:
                    prev_history_loss_avg = curr_loss_history
                elif curr_loss_history - prev_history_loss_avg > 10**6:
                    print("Huge gap in the loss")
                    save_path = manage_issues.save()
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
                distributed_test_step(elt)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=step_int)
            print("Epoch {:03d}: Train Loss: {:.3f} Val Loss: {:03f}".format(
                epoch, epoch_loss_avg.result(), test_loss.result()))
            # Generate some samples and visualize them on tensoboard
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

    # Saving the last variables
    manager.save()
    print("Model Saved at {}".format(save_path))

    # Training Time
    t1 = time.time()
    training_time = t1 - t0
    print("Training time: ", np.round(training_time, 2), ' seconds')

    log_file.close()


if __name__ == '__main__':
    main()
