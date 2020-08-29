import tensorflow as tf
import numpy as np
tfk = tf.keras


def get_sigmas(sigma1, sigmaL, num_classes):

    sigmas = np.exp(np.linspace(np.log(sigma1), np.log(sigmaL), num=num_classes))
    return sigmas


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
            if return_arr:
                x_arr = np.concatenate((x_arr, tf.expand_dims(x_mod, axis=0).numpy()), axis=0)

    if return_arr:
        return x_arr
    else:
        return x_mod.numpy()
