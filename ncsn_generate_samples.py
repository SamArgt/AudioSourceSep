import numpy as np
import tensorflow as tf
from ncsn.utils import *
from train_utils import *
import argparse
import time
import os
import tensorflow_addons as tfa
tfk = tf.keras


def setUp_optimizer(args):

    if args.optimizer == 'adam':
        optimizer = tfk.optimizers.Adam()
    elif args.optimizer == 'adamax':
        optimizer = tfk.optimizers.Adamax()
    else:
        raise ValueError("optimizer argument should be adam or adamax")

    return optimizer


def main(args):

    if args.config is not None:
        new_args = get_config(args.config)
        new_args.dataset = args.dataset
        new_args.filename = args.filename
        new_args.RESTORE = args.RESTORE
        new_args.n_samples = args.n_samples
        args = new_args

    # Print parameters
    print("SAMPLING PARAMETERS")
    params_dict = vars(args)
    template = '\t '
    for k, v in params_dict.items():
        template += '{} = {} \n\t '.format(k, v)
    print(template)
    print("_" * 100)

    sigmas_np = get_sigmas(args.sigma1, args.sigmaL, args.num_classes)
    sigmas_tf = tf.constant(sigmas_np, dtype=tf.float32)

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
    if args.version == 'v2':
        model = get_uncompiled_model_v2(args, sigmas=sigmas_tf)
    else:
        model = get_uncompiled_model(args)
    optimizer = setUp_optimizer(args)
    if args.ema:
        optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.999)
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
                                     n_steps_each=args.T, step_lr=args.step_lr,
                                     return_arr=args.return_last_point)
    x_arr = post_processing(x_arr)
    print("Done. Duration: {} seconds".format(round(time.time() - t0, 2)))
    print("Shape: {}".format(x_arr.shape))
    if args.filename is None:
        args.filename, ckpt_name = os.path.split(abs_restore_path)

        args.filename = os.path.join(args.filename, "generated_samples" + '_' + ckpt_name)
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

    # Output and Restore Directory
    parser.add_argument('--filename', type=str, default=None,
                        help='filename for savings')

    # dataset parameters
    parser.add_argument('--dataset', type=str, default="melspec",
                        help="mnist or cifar10 or directory to tfrecords")

    parser.add_argument("--n_samples", type=int, default=32,
                        help="Number of samples to generate")

    # config
    parser.add_argument('--config', type=str, help='path to the config file. Overwrite all other parameters below')

    # NCSNv2
    parser.add_argument('--version', type=str, help='Version of NCSN', default='v2')
    parser.add_argument('--ema', action='store_true', help="Use Exponential Moving Average")

    # Sampling parameters
    parser.add_argument("--T", type=int, default=100,
                        help="Number of step for each sigma in the Langevin Dynamics")
    parser.add_argument("--step_lr", type=float, default=2e-5,
                        help="learning rate in the lengevin dynamics")
    parser.add_argument("--return_last_point", action="store_false",
                        help="Either to return array of every steps or just the last point")

    # Spectrograms Parameters
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--scale", type=str, default='dB',
                        help="power or dB")

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
