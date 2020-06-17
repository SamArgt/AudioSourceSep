import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flow_models import flow_builder
from pipeline import data_loader
import argparse
import time
import os
import sys
import shutil
import datetime
import io
import matplotlib.pyplot as plt
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def setUp_optimizer(args):
    lr = args.learning_rate
    if args.optimizer == 'adam':
        optimizer = tfk.optimizers.Adam(lr=lr, clipvalue=args.clipvalue, clipnorm=args.clipnorm)
    elif args.optimizer == 'adamax':
        optimizer = tfk.optimizers.Adamax(lr=lr)
    else:
        raise ValueError("optimizer argument should be adam or adamax")
    return optimizer


def setUp_checkpoint(model, optimizer):
    # Checkpoint object
    ckpt = tf.train.Checkpoint(
        variables=model.variables, optimizer=optimizer)
    return ckpt


def restore_checkpoint(ckpt, restore_path, model, optimizer, latest=True):
    if latest:
        checkpoint_restore_path = tf.train.latest_checkpoint(restore_path)
        assert restore_path is not None, restore_path
    else:
        checkpoint_restore_path = restore_path
    # Restore weights if specified
    status = ckpt.restore(checkpoint_restore_path)
    status.assert_existing_objects_matched()

    return ckpt


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


def image_grid(n_display, x, y, z, separation=True):
    # Create a figure to contain the plot.
    f, axes = plt.subplots(nrows=n_display, ncols=3, figsize=(6, 8))
    for i in range(n_display):
        ax1, ax2, ax3 = axes[i]
        ax1.imshow(x[i], cmap=plt.cm.binary)
        ax2.imshow(y[i], cmap=plt.cm.binary)
        ax3.imshow(z[i], cmap=plt.cm.binary)
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()

    if separation:
        title = "Separation: Mixture = Component 1 + Component 2"
    else:
        title = "Mixing: Component 1 + Component 2 = Mixture"
    f.suptitle(title)
    return f


@tf.function
def compute_grad_logprob(X, model, debug=False):
    with tf.GradientTape() as tape:
        tape.watch(X)
        loss = model.log_prob(X)
    gradients = tape.gradient(loss, X)

    return gradients


def basis_inner_loop(mixed, x1, x2, model1, model2, sigma, n_mixed,
                     sigmaL=0.01, delta=3e-5, T=100, dataset="mnist", debug=True):

    if dataset == 'mnist':
        data_shape = [n_mixed, 32, 32, 1]
    elif dataset == 'cifar10':
        data_shape == [n_mixed, 32, 32, 3]

    eta = float(delta * (sigma / sigmaL) ** 2)
    lambda_recon = 1.0 / (sigma ** 2)
    for t in range(T):
        epsilon1 = tf.math.sqrt(2. * eta) * tf.random.normal(data_shape)
        epsilon2 = tf.math.sqrt(2. * eta) * tf.random.normal(data_shape)

        grad_logprob1 = compute_grad_logprob(x1, model1)
        grad_logprob2 = compute_grad_logprob(x2, model2)

        if debug:
            assert grad_logprob1.shape == x1.shape
            assert bool(tf.math.is_nan(grad_logprob1).numpy().any()) is False, (sigma, t)
            assert grad_logprob2.shape == x2.shape
            assert bool(tf.math.is_nan(grad_logprob2).numpy().any()) is False, (sigma, t)

        x1 = x1 + eta * (grad_logprob1 - lambda_recon * (x1 + x2 - 2. * mixed)) + epsilon1
        x2 = x2 + eta * (grad_logprob2 - lambda_recon * (x1 + x2 - 2. * mixed)) + epsilon2

    if debug:
        assert bool(tf.math.is_nan(x1).numpy().any()) is False, (sigma, t)
        assert bool(tf.math.is_nan(x2).numpy().any()) is False, (sigma, t)

    return x1, x2


def basis_outer_loop(mixed, x1, x2, model, optimizer, restore_dict,
                     ckpt, args, train_summary_writer):

    step = 0
    for sigma, restore_path in restore_dict.items():
        restore_checkpoint(ckpt, restore_path, model, optimizer)
        print("Model at noise level {} restored from {}".format(sigma, restore_path))
        model1 = model2 = model

        x1, x2 = basis_inner_loop(mixed, x1, x2, model1, model2, sigma, args.n_mixed, sigmaL=args.sigmaL,
                                  delta=3e-5, T=args.T, dataset=args.dataset, debug=args.debug)
        step += 1

        with train_summary_writer.as_default():
            if args.n_mixed > 5:
                n_display = 5
            else:
                n_display = args.n_mixed

            sample_mix = mixed.numpy()[:n_display].reshape((n_display, 32, 32))
            sample_x1 = x1.numpy()[:n_display].reshape((n_display, 32, 32))
            sample_x2 = x2.numpy()[:n_display].reshape((n_display, 32, 32))
            figure = image_grid(n_display, sample_mix, sample_x1, sample_x2, separation=True)
            tf.summary.image("Components", plot_to_image(figure),
                             max_outputs=10, step=step)

        print("inner loop done")
        print("_" * 100)

    return x1, x2


def main(args):

    # noise conditionned models
    abs_restore_path = os.path.abspath(args.RESTORE)
    sigmas = np.logspace(np.log(args.sigma1) / np.log(10), np.log(args.sigmaL) / np.log(10), num=args.n_sigmas)
    restore_dict = {sigma: os.path.join(abs_restore_path, "sigma_" + str(round(sigma, 2)), "tf_ckpts") for sigma in sigmas}

    if args.debug:
        for k, v in restore_dict.items():
            print("{}: {}".format(round(k, 2), v))

    try:
        os.mkdir(args.output)
        os.chdir(args.output)
    except FileExistsError:
        os.chdir(args.output)

    log_file = open('out.log', 'w')
    if args.debug is False:
        sys.stdout = log_file

    # set up tensorboard
    train_summary_writer, test_summary_writer = setUp_tensorboard()

    params_dict = vars(args)
    template = 'BASIS Separation \n\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)
    with train_summary_writer.as_default():
        tf.summary.text(name='Parameters',
                        data=tf.constant(template), step=0)

    # get mixture
    mixed, x1, x2, gt1, gt2, minibatch = data_loader.get_mixture(dataset=args.dataset, n_mixed=args.n_mixed,
                                                                 use_logit=args.use_logit, alpha=args.alpha, noise=None,
                                                                 mirrored_strategy=None)
    with train_summary_writer.as_default():
        if args.n_mixed > 5:
            n_display = 5
        else:
            n_display = args.n_mixed

        sample_mix = mixed.numpy()[:n_display].reshape((n_display, 32, 32))
        sample_gt1 = gt1.numpy()[:n_display].reshape((n_display, 32, 32))
        sample_gt2 = gt2.numpy()[:n_display].reshape((n_display, 32, 32))
        figure = image_grid(n_display, sample_gt1, sample_gt2, sample_mix, separation=False)
        tf.summary.image("Originals", plot_to_image(figure),
                         max_outputs=1, step=0)

    # build model
    model = flow_builder.build_flow(minibatch, L=args.L, K=args.K, n_filters=args.n_filters, dataset=args.dataset,
                                    l2_reg=args.l2_reg, mirrored_strategy=None)
    # set up optimizer
    optimizer = setUp_optimizer(args)

    # checkpoint
    ckpt = setUp_checkpoint(model, optimizer)

    # run BASIS separation
    t0 = time.time()
    x1, x2 = basis_outer_loop(mixed, x1, x2, model, optimizer, restore_dict,
                              ckpt, args, train_summary_writer)
    t1 = time.time()
    print("Duration: {} seconds".format(round(t1 - t0, 3)))

    # Save results
    x1 = x1.numpy()
    x2 = x2.numpy()
    np.savez('results', x1=x1, x2=x2, gt1=gt1, gt2=gt2, mixed=mixed)

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='BASIS Separatation')
    parser.add_argument('RESTORE', type=str, default=None,
                        help='directory of saved models')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10")
    parser.add_argument('--output', type=str, default='basis_sep',
                        help='output dirpath for savings')
    parser.add_argument('--debug', action="store_true")

    # BASIS hyperparameters
    parser.add_argument("--T", type=int, default=100,
                        help="Number of iteration in the inner loop")
    parser.add_argument('--n_mixed', type=int, default=10,
                        help="number of mixture to separate")
    parser.add_argument('--sigma1', type=float, default=1.0)
    parser.add_argument('--sigmaL', type=float, default=0.01)
    parser.add_argument('--n_sigmas', type=float, default=10)

    # Model hyperparameters
    parser.add_argument('--L', default=2, type=int,
                        help='Depth level')
    parser.add_argument('--K', type=int, default=16,
                        help="Number of Step of Flow in each Block")
    parser.add_argument('--n_filters', type=int, default=256,
                        help="number of filters in the Convolutional Network")
    parser.add_argument('--l2_reg', type=float, default=None,
                        help="L2 regularization for the coupling layer")
    parser.add_argument("--learntop", action="store_true",
                        help="learnable prior distribution")

    # Optimization parameters
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
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

    args = parser.parse_args()

    main(args)
