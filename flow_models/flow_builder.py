from .flow_glow import *
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def build_flow(L=3, K=32, n_filters=512, dataset='mnist', l2_reg=None, mirrored_strategy=None):
    tfk.backend.clear_session()
    # Set flow parameters
    if dataset == 'mnist':
        data_shape = (32, 32, 1)
    elif dataset == 'cifar10':
        data_shape = (32, 32, 3)
    else:
        raise ValueError("dataset should be mnist or cifar10")

    if L == 2:
        base_distr_shape = [data_shape[0] // 4,
                            data_shape[1] // 4, data_shape[2] * 16]
    elif L == 3:
        base_distr_shape = [data_shape[0] // 8,
                            data_shape[1] // 8, data_shape[2] * 64]
    else:
        raise ValueError("L should be 2 or 3")

    shift_and_log_scale_layer = ShiftAndLogScaleResNet

    # Build Flow and Optimizer
    if mirrored_strategy is not None:
        with mirrored_strategy.scope():
            if L == 2:
                bijector = GlowBijector_2blocks(K, data_shape,
                                                shift_and_log_scale_layer,
                                                n_filters, minibatch, **{'l2_reg': l2_reg})
            elif L == 3:
                bijector = GlowBijector_3blocks(K, data_shape,
                                                shift_and_log_scale_layer,
                                                n_filters, minibatch, **{'l2_reg': l2_reg})
            inv_bijector = tfb.Invert(bijector)
            flow = tfd.TransformedDistribution(tfd.Normal(
                0., 1.), inv_bijector, event_shape=base_distr_shape)

    else:
        if L == 2:
            bijector = flow_glow.GlowBijector_2blocks(K, data_shape,
                                                      shift_and_log_scale_layer,
                                                      n_filters, minibatch, **{'l2_reg': l2_reg})
        elif L == 3:
            bijector = flow_glow.GlowBijector_3blocks(K, data_shape,
                                                      shift_and_log_scale_layer,
                                                      n_filters, minibatch, **{'l2_reg': l2_reg})
        inv_bijector = tfb.Invert(bijector)
        flow = tfd.TransformedDistribution(tfd.Normal(
            0., 1.), inv_bijector, event_shape=base_distr_shape)

    return flow
