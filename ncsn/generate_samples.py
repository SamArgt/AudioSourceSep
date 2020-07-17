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


def anneal_langevin_dynamics(data_shape, model, n_samples, sigmas, n_steps_each=100,
                             step_lr=0.00002, return_arr=True, training=False):
    """
    Anneal Langevin dynamics
    """
    x_mod = tf.random.uniform([n_samples] + list(data_shape))
    if return_arr:
        x_arr = tf.expand_dims(x_mod, axis=0).numpy()
    for i, sigma in enumerate(sigmas):
        print("Sigma = {} ({} / {})".format(sigma, i + 1, len(sigmas)))
        labels = tf.ones(n_samples, dtype=tf.int32) * i
        step_size = tf.constant(step_lr * (sigma / sigmas[-1]) ** 2, dtype=tf.float32)
        for s in range(n_steps_each):
            if ((s + 1) % (n_steps_each // 10) == 0):
                print("Step {} / {}".format(s + 1, n_steps_each))
                if return_arr:
                    x_arr = np.concatenate((x_arr, tf.expand_dims(x_mod, axis=0).numpy()), axis=0)
            noise = tf.random.normal([n_samples] + list(data_shape)) * tf.math.sqrt(step_size * 2)
            grad = model([x_mod, labels], training=training)
            x_mod = x_mod + step_size * grad + noise

    if return_arr:
        return x_arr
    else:
        return x_mod.numpy()


def setUp_optimizer(args):
    lr = args.learning_rate

    if args.optimizer == 'adam':
        optimizer = tfk.optimizers.Adam(lr=lr, clipvalue=args.clipvalue, clipnorm=args.clipnorm)
    elif args.optimizer == 'adamax':
        optimizer = tfk.optimizers.Adamax(lr=lr)
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

    # Restore Model
    abs_restore_path = os.path.abspath(args.RESTORE)
    model = get_uncompiled_model(args)
    optimizer = setUp_optimizer(args)
    model.compile(optimizer=optimizer, loss=tfk.losses.MeanSquaredError())
    model.load_weights(abs_restore_path)
    print("Weights loaded")

    print("Start Generating {} samples....".format(args.n_samples))
    t0 = time.time()
    x_arr = anneal_langevin_dynamics(args.data_shape, model, args.n_samples, sigmas_np,
                                     n_steps_each=args.n_steps_each, step_lr=args.step_lr,
                                     return_arr=args.return_last_point, training=args.training)
    if args.use_logit:
        x_arr = 1. / (1 - np.exp(-x_arr))
        x_arr = (x_arr - args.alpha) / (1 - 2 * args.alpha)

    print("Done. Duration: {} seconds".format(round(time.time() - t0, 2)))
    print("Shape: {}".format(x_arr.shape))
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
    parser.add_argument("--training", action='store_true',
                        help="Either to use inference mode or training mode in the network")

    # dataset parameters
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="mnist or cifar10 or directory to tfrecords")
    # Spectrograms Parameters
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)

    # Output and Restore Directory
    parser.add_argument('--filename', type=str, default='ncsn_generated_samples',
                        help='filename for savings')

    # Model hyperparameters
    parser.add_argument("--n_filters", type=int, default=64,
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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipvalue', type=float, default=None,
                        help="Clip value for Adam optimizer")
    parser.add_argument('--clipnorm', type=float, default=None,
                        help='Clip norm for Adam optimize')

    args = parser.parse_args()

    main(args)
