import tensorflow as tf
import numpy as np
from . import score_network, score_network_v2
tfk = tf.keras


def get_sigmas(sigma1, sigmaL, num_classes, progression='geometric'):
    if progression == 'geometric':
        sigmas = np.exp(np.linspace(np.log(sigma1), np.log(sigmaL), num=num_classes))
    elif progression == 'logarithmic':
        sigmas = np.logspace(np.log(sigma1) / np.log(10), np.log(sigmaL) / np.log(10), num=num_classes)
    else:
        raise ValueError('progression should be geometric or logarithmic')
    return sigmas.astype(np.float32)


def anneal_langevin_dynamics(x_mod, data_shape, model, n_samples, sigmas, n_steps_each=100, step_lr=2e-5, return_arr=False, verbose=False):
    """
    Anneal Langevin dynamics
    """
    if return_arr:
        x_arr = tf.expand_dims(x_mod, axis=0).numpy()
    for i, sigma in enumerate(sigmas):
        if verbose:
            print("Sigma = {} ({} / {})".format(sigma, i + 1, len(sigmas)))
        labels = tf.ones(n_samples, dtype=tf.int32) * i
        step_size = tf.constant(step_lr * (sigma / sigmas[-1]) ** 2, dtype=tf.float32)
        for s in range(n_steps_each):
            noise = tf.random.normal([n_samples] + list(data_shape)) * tf.math.sqrt(step_size * 2)
            grad = model([x_mod, labels], training=True)
            x_mod = x_mod + step_size * grad + noise
        if return_arr:
            x_arr = np.concatenate((x_arr, tf.expand_dims(x_mod, axis=0).numpy()), axis=0)

    if return_arr:
        return x_arr
    else:
        return x_mod.numpy()


def get_uncompiled_model(args, name="ScoreNetwork"):
    # inputs
    perturbed_X = tfk.Input(shape=args.data_shape, dtype=tf.float32, name="perturbed_X")
    sigma_idx = tfk.Input(shape=[], dtype=tf.int32, name="sigma_idx")
    # outputs
    outputs = score_network.CondRefineNetDilated(args.data_shape, args.n_filters,
                                                 args.num_classes, args.use_logit)([perturbed_X, sigma_idx])
    # model
    model = tfk.Model(inputs=[perturbed_X, sigma_idx], outputs=outputs, name=name)

    return model


def get_uncompiled_model_v2(args, sigmas, name="ScoreNetworkv2"):
    # inputs
    perturbed_X = tfk.Input(shape=args.data_shape, dtype=tf.float32, name="perturbed_X")
    sigma_idx = tfk.Input(shape=[], dtype=tf.int32, name="sigma_idx")
    # outputs
    outputs = score_network_v2.RefineNetDilated(args.data_shape, args.n_filters,
                                                sigmas, args.use_logit)([perturbed_X, sigma_idx])
    # model
    model = tfk.Model(inputs=[perturbed_X, sigma_idx], outputs=outputs, name=name)

    return model
