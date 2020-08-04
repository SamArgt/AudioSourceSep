import numpy as np
import tensorflow as tf
import score_network
import argparse
import time
import os
tfk = tf.keras


def get_uncompiled_model(args):
    # inputs
    perturbed_X = tfk.Input(shape=args.data_shape, dtype=tf.float32, name="perturbed_X")
    sigma_idx = tfk.Input(shape=[], dtype=tf.int32, name="sigma_idx")
    # outputs
    outputs = score_network.CondRefineNetDilated(args.data_shape, args.n_filters,
                                                 args.num_classes, args.use_logit)([perturbed_X, sigma_idx])
    # model
    model = tfk.Model(inputs=[perturbed_X, sigma_idx], outputs=outputs, name="ScoreNetwork")

    return model


class CustomLoss(tfk.losses.Loss):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def __call__(self, scores, target, sample_weight=None):
        loss = (1 / 2.) * tf.reduce_sum(tf.square(scores - target), axis=[1, 2, 3])
        if sample_weight is not None:
            return tf.reduce_mean(loss * sample_weight)
        else:
            return tf.reduce_mean(loss)


def anneal_langevin_dynamics(x_mod, data_shape, model, n_samples, sigmas, n_steps_each=100, step_lr=2e-5, return_arr=False):
    """
    Anneal Langevin dynamics
    """
    if return_arr:
        x_arr = tf.expand_dims(x_mod, axis=0).numpy()
    for i, sigma in enumerate(sigmas):
        print("Sigma = {} ({} / {})".format(sigma, i + 1, len(sigmas)))
        labels = tf.ones(n_samples, dtype=tf.int32) * i
        step_size = tf.constant(step_lr * (sigma / sigmas[-1]) ** 2, dtype=tf.float32)
        for s in range(n_steps_each):
            noise = tf.random.normal([n_samples] + list(data_shape)) * tf.math.sqrt(step_size * 2)
            grad = model([x_mod, labels], training=True)
            x_mod = x_mod + step_size * grad + noise
            if ((s + 1) % (n_steps_each // 10) == 0):
                print("Step {} / {}".format(s + 1, n_steps_each))
                if return_arr:
                    x_arr = np.concatenate((x_arr, tf.expand_dims(x_mod, axis=0).numpy()), axis=0)

    if return_arr:
        return x_arr
    else:
        return x_mod.numpy()


def setUp_optimizer(args):

    if args.optimizer == 'adam':
        optimizer = tfk.optimizers.Adam()
    elif args.optimizer == 'adamax':
        optimizer = tfk.optimizers.Adamax()
    else:
        raise ValueError("optimizer argument should be adam or adamax")

    return optimizer


def main(args):

    # Print parameters
    print("SAMPLING PARAMETERS")
    params_dict = vars(args)
    template = '\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)
    print("_" * 100)

    sigmas_np = np.logspace(np.log(args.sigma1) / np.log(10),
                            np.log(args.sigmaL) / np.log(10),
                            num=args.num_classes)

    # data paramaters
    if args.dataset == 'mnist':
        args.data_shape = [32, 32, 1]
        args.data_type = "image"
        args.minval = 0.
        args.maxval = 256.
    elif args.dataset == 'cifar10':
        args.data_shape = [32, 32, 3]
        args.data_type = "image"
        args.minval = 0.
        args.maxval = 256.
    else:
        args.data_shape = [args.height, args.width, 1]
        args.data_type = "melspec"
        if args.scale == 'power':
            args.minval = 1e-10
            args.maxval = 100.
        elif args.scale == 'dB':
            args.minval = -100.
            args.maxval = 20.
        else:
            raise ValueError("scale should be 'power' or 'dB'")

    def post_processing(x):
        if args.use_logit:
            x = 1. / (1. + np.exp(-x))
            x = (x - args.alpha) / (1. - 2. * args.alpha)
        x = x * (args.maxval - args.minval) + args.minval
        if args.data_type == 'image':
            x = np.clip(x, 0., 255.)
            x = np.round(x, decimals=0).astype(int)
        else:
            x = np.clip(x, args.minval, args.maxval)
        return x

    # Restore Model
    abs_restore_path = os.path.abspath(args.RESTORE)
    model = get_uncompiled_model(args)
    optimizer = setUp_optimizer(args)
    ckpt = tf.train.Checkpoint(variables=model.variables, optimizer=optimizer)
    status = ckpt.restore(abs_restore_path)
    status.assert_existing_objects_matched()
    print("Weights loaded")

    print("Start Generating {} samples....".format(args.n_samples))
    t0 = time.time()
    x_mod = tf.random.uniform(shape=[args.n_samples] + args.data_shape)
    if args.use_logit:
        x_mod = (1. - 2 * args.alpha) * x_mod + args.alpha
        x_mod = tf.math.log(x_mod) - tf.math.log(1. - x_mod)
    x_arr = anneal_langevin_dynamics(x_mod, args.data_shape, model, args.n_samples, sigmas_np,
                                     n_steps_each=args.n_steps_each, step_lr=args.step_lr,
                                     return_arr=args.return_last_point)
    x_arr = post_processing(x_arr)
    print("Done. Duration: {} seconds".format(round(time.time() - t0, 2)))
    print("Shape: {}".format(x_arr.shape))
    if args.filename is None:
        args.filename = os.path.split(abs_restore_path)[0]
        args.filename = os.path.join(args.filename, "generated_samples")
    try:
        np.save(args.filename, x_arr)
        print("Generated Samples saved at {}".format(args.filename + ".npy"))
    except FileNotFoundError:
        np.save("generated_samples", x_arr)
        print("Generated Samples saved at {}".format("generated_samples.npy"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample from NCSN model')

    parser.add_argument('RESTORE', type=str, default=None,
                        help='directory of saved weights')

    # Sampling parameters
    parser.add_argument("--n_samples", type=int, default=32,
                        help="Number of samples to generate")
    parser.add_argument("--n_steps_each", type=int, default=100,
                        help="Number of step for each sigma in the Langevin Dynamics")
    parser.add_argument("--step_lr", type=float, default=2e-5,
                        help="learning rate in the lengevin dynamics")
    parser.add_argument("--return_last_point", action="store_false",
                        help="Either to return array of every steps or just the last point")

    # dataset parameters
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10 or directory to tfrecords")
    # Spectrograms Parameters
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--scale", type=str, default='dB',
                        help="power or dB")

    # Output and Restore Directory
    parser.add_argument('--filename', type=str, default=None,
                        help='filename for savings')

    # Model hyperparameters
    parser.add_argument("--n_filters", type=int, default=192,
                        help='number of filters in the Score Network')
    parser.add_argument("--sigma1", type=float, default=1.)
    parser.add_argument("--sigmaL", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=10)

    # Preprocessing parameters
    parser.add_argument("--use_logit", action="store_true")
    parser.add_argument("--alpha", type=float, default=1e-6)

    # Optimization parameters
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="adam or adamax")

    args = parser.parse_args()

    main(args)
