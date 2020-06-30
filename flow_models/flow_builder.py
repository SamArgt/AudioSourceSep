from .flow_glow import *
from .flow_tfk_layers import *
from .flow_tfp_bijectors import *
from .flow_flowpp import *
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def build_glow(minibatch, data_shape, L=3, K=32, n_filters=512, learntop=True, l2_reg=None,
               mirrored_strategy=None, preprocessing='melspec'):
    tfk.backend.clear_session()

    if L == 2:
        base_distr_shape = [data_shape[0] // 4,
                            data_shape[1] // 4, data_shape[2] * 16]
        bijector_cls = GlowBijector_2blocks
    elif L == 3:
        base_distr_shape = [data_shape[0] // 8,
                            data_shape[1] // 8, data_shape[2] * 64]
        bijector_cls = GlowBijector_3blocks
    else:
        raise ValueError("L should be 2 or 3")

    shift_and_log_scale_layer = ShiftAndLogScaleResNet

    # Build Flow and Optimizer
    if mirrored_strategy is not None:
        with mirrored_strategy.scope():

            flow_bijector = bijector_cls(K, data_shape,
                                         shift_and_log_scale_layer,
                                         n_filters, minibatch, **{'l2_reg': l2_reg})
            if preprocessing == "melspec":
                prepocessing_bijector = SpecPreprocessing()
                flow_bijector = tfb.Chain([flow_bijector, prepocessing_bijector])

            inv_bijector = tfb.Invert(flow_bijector)

            if learntop:
                prior_distribution = tfd.Independent(tfd.MultivariateNormalDiag(
                    loc=tf.Variable(tf.zeros(base_distr_shape), name='loc'),
                    scale_diag=tfp.util.TransformedVariable(
                        tf.ones(base_distr_shape),
                        bijector=tfb.Exp()),
                    name='scale'),
                    reinterpreted_batch_ndims=2,
                    name='learnable_mvn_scaled_identity')
                flow = tfd.TransformedDistribution(
                    prior_distribution, inv_bijector)
            else:
                flow = tfd.TransformedDistribution(tfd.Normal(
                    0., 1.), inv_bijector, event_shape=base_distr_shape)

    else:
        flow_bijector = bijector_cls(K, data_shape,
                                     shift_and_log_scale_layer,
                                     n_filters, minibatch, **{'l2_reg': l2_reg})
        if preprocessing == "melspec":
            prepocessing_bijector = SpecPreprocessing()
            flow_bijector = tfb.Chain([flow_bijector, prepocessing_bijector])

        inv_bijector = tfb.Invert(flow_bijector)

        if learntop:
            prior_distribution = tfd.Independent(tfd.MultivariateNormalDiag(
                loc=tf.Variable(tf.zeros(base_distr_shape), name='loc'),
                scale_diag=tfp.util.TransformedVariable(
                    tf.ones(base_distr_shape),
                    bijector=tfb.Exp()),
                name='scale'),
                reinterpreted_batch_ndims=2,
                name='learnable_mvn_scaled_identity')
            flow = tfd.TransformedDistribution(
                prior_distribution, inv_bijector)
        else:
            flow = tfd.TransformedDistribution(tfd.Normal(
                0., 1.), inv_bijector, event_shape=base_distr_shape)

    return flow


def build_flowpp(minibatch, data_shape, n_components=32, n_blocks_flow=10,
                 n_blocks_dequant=2, filters=96, dropout_p=0., heads=4,
                 mirrored_strategy=None):

    base_distr_shape = [data_shape[0] // 2,
                        data_shape[1] // 2,
                        data_shape[2] * 4]

    if mirrored_strategy is not None:
        with mirrored_strategy.scope():

            dequant_flow = DequantFlowpp(data_shape, n_components=n_components,
                                         n_blocks=n_blocks_dequant, filters=filters,
                                         dropout_p=dropout_p, heads=heads)

            flowpp_cifar10 = Flowpp_cifar10(data_shape, minibatch, n_components=n_components,
                                            n_blocks=n_blocks_flow, filters=filters,
                                            dropout_p=dropout_p, heads=heads)

            bijector = tfb.Chain([flowpp_cifar10, dequant_flow])
            inv_bijector = tfb.Invert(bijector)

            flow = tfd.TransformedDistribution(tfd.Normal(
                0., 1.), inv_bijector, event_shape=base_distr_shape)

    else:
        dequant_flow = DequantFlowpp(data_shape, n_components=n_components,
                                     n_blocks=n_blocks_dequant, filters=filters,
                                     dropout_p=dropout_p, heads=heads)

        flowpp_cifar10 = Flowpp_cifar10(data_shape, minibatch, n_components=n_components,
                                        n_blocks=n_blocks_flow, filters=filters,
                                        dropout_p=dropout_p, heads=heads)

        bijector = tfb.Chain([flowpp_cifar10, dequant_flow])
        inv_bijector = tfb.Invert(bijector)

        flow = tfd.TransformedDistribution(tfd.Normal(
            0., 1.), inv_bijector, event_shape=base_distr_shape)

    return flow
