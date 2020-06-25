import tensorflow as tf
import tensorflow_probability as tfp
from .flow_tfp_bijectors import *
from .flow_tfk_layers import *
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


class FlowppCouplingLayer(tfp.bijectors.Bijector):

    def __init__(self, input_shape, minibatch, split='channel', split_state=0, n_components=32, n_blocks=10, filters=96,
                 dropout_p=0., heads=4, context=False, name="FlowppCouplingLayer"):

        super(FlowppCouplingLayer, self).__init__(forward_min_event_ndims=3)

        self.actnorm = ActNorm(input_shape, minibatch,
                               normalize='all', name=name + "/actnorm")
        self.inv1x1conv = Invertible1x1Conv(
            input_shape, name=name + '/inv1x1conv')
        self.mixLogCdfAttnCoupling = MixLogisticCDFAttnCoupling(input_shape, split=split, split_state=split_state,
                                                                n_components=n_components, n_blocks=n_blocks, filters=filters,
                                                                dropout_p=dropout_p, heads=heads, context=context,
                                                                name=name + "/mixLogCdfAttnCoupling")

    def _forward(self, x, context=None):
        y = self.actnorm.forward(x)
        y = self.inv1x1conv.forward(y)
        return self.mixLogCdfAttnCoupling.forward(x, context=context)

    def _inverse(self, y, context=None):
        x = self.mixLogCdfAttnCoupling.inverse(y, context=context)
        x = self.inv1x1conv.inverse(x)
        return self.actnorm.inverse(y)

    def _forward_log_det_jacobian(self, x, context=None):
        log_det = self.actnorm.forward_log_det_jacobian(x, event_ndims=3)
        y = self.actnorm.forward(x)
        log_det += self.inv1x1conv.forward_log_det_jacobian(y, event_ndims=3)
        y = self.inv1x1conv.forward(y)
        return log_det + self.mixLogCdfAttnCoupling.forward_log_det_jacobian(y, context=context, event_ndims=3)


class FlowppBlock(tfp.bijectors.Bijector):

    def __init__(self, input_shape, minibatch, n_layers, split="channel", n_components=32, n_blocks=10, filters=96,
                 dropout_p=0., heads=4, context=False, name="FlowppBlock"):

        super(FlowppBlock, self).__init__(forward_min_event_ndims=3)

        self.coupling_layers = []
        split_state = 0
        minibatch_updated = minibatch
        for i in range(n_layers):

            coupling_layer = FlowppCouplingLayer(input_shape, minibatch_updated, split=split,
                                                 split_state=split_state, n_components=n_components,
                                                 n_blocks=n_blocks, filters=filters,
                                                 dropout_p=dropout_p, heads=heads, context=context,
                                                 name=name + "/FlowppCouplingLayer" + str(i + 1))
            minibatch_updated = coupling_layer.forward(minibatch_updated)
            self.coupling_layers.append(coupling_layer)
            split_state = split_state + 1 % 2

    def _forward(self, x, context=None):
        for coupling_layer in self.coupling_layers:
            x = coupling_layer.forward(x, context=context)
        return x

    def _inverse(self, y, context=None):
        for coupling_layer in reversed(self.coupling_layers):
            y = coupling_layer.inverse(y, context=context)
        return y

    def _forward_log_det_jacobian(self, x, context=None):
        for i, coupling_layer in enumerate(self.coupling_layers):
            x = coupling_layer.forward(x, context=context)
            if i == 0:
                log_det = coupling_layer.forward_log_det_jacobian(x, context=context, event_ndims=3)
            else:
                log_det += coupling_layer.forward_log_det_jacobian(x, context=context, event_ndims=3)

        return log_det


class Flowpp_cifar10(tfp.bijectors.Bijector):

    def __init__(self, input_shape, minibatch, n_components=32, n_blocks=10, filters=96,
                 dropout_p=0., heads=4, name="flowpp_cifar10"):

        super(Flowpp_cifar10, self).__init__(forward_min_event_ndims=3)

        self.preprocessing = Preprocessing(input_shape, use_logit=True, uniform_noise=False, alpha=0.05)
        minibatch_updated = self.preprocessing.forward(minibatch)

        self.flow_block1 = FlowppBlock(input_shape, minibatch_updated, 4, split="checkerboard", n_components=n_components,
                                       n_blocks=n_blocks, filters=filters,
                                       dropout_p=dropout_p, heads=heads,
                                       name=name + "/flowBlock1")
        minibatch_updated = self.flow_block1.forward(minibatch)

        self.squeeze = Squeeze(input_shape)
        minibatch_updated = self.squeeze.forward(minibatch_updated)
        event_shape_out = self.squeeze.event_shape_out

        self.flow_block2 = FlowppBlock(event_shape_out, minibatch_updated, 2, split="channel",
                                       n_components=n_components,
                                       n_blocks=n_blocks, filters=filters,
                                       dropout_p=dropout_p, heads=heads,
                                       name=name + "/flowBlock2")

        minibatch_updated = self.flow_block2.forward(minibatch_updated)

        self.flow_block3 = FlowppBlock(event_shape_out, minibatch_updated,
                                       3, split="checkerboard", n_components=n_components,
                                       n_blocks=n_blocks, filters=filters,
                                       dropout_p=dropout_p, heads=heads,
                                       name=name + "/flowBlock3")

        self.bijector = tfb.Chain(
            [self.flow_block3, self.flow_block2, self.squeeze, self.flow_block1, self.preprocessing])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, y):
        return self.bijector.inverse(y)

    def _forward_log_det_jacobian(self, x):
        return self.bijector.forward_log_det_jacobian(x, event_ndims=3)

    def _forward_event_shape_tensor(self, input_shape):
        H, W, C = input_shape
        return (H // 2, W // 2, C * 4)

    def _forward_event_shape(self, input_shape):
        H, W, C = input_shape
        return (H // 2, W // 2, C * 4)

    def _inverse_event_shape_tensor(self, output_shape):
        H, W, C = output_shape
        return (H * 2, W * 2, C // 4)

    def _inverse_event_shape(self, output_shape):
        H, W, C = output_shape
        return (H * 2, W * 2, C // 4)


class DequantFlowpp(tfp.bijectors.Bijector):

    def __init__(self, input_shape, n_components=32, n_blocks=2, filters=96,
                 dropout_p=0., heads=4, name="dequant_flowpp"):

        super(DequantFlowpp, self).__init__(forward_min_event_ndims=3)

        self.H, self.W, self.C = input_shape
        reshaped_shape = [self.H, self.W // 2, 2 * self.C]
        self.processor = ShallowProcessor(reshaped_shape, filters=32, dropout_p=dropout_p)
        eps = tf.random.normal([4] + list(input_shape))
        self.bijector = FlowppBlock(input_shape, eps, 4, split="checkerboard",
                                    n_components=n_components, n_blocks=n_blocks, filters=filters,
                                    dropout_p=dropout_p, heads=heads, context=True,
                                    name=name + "flow_block")
        self.log_det_eps = None

    def _forward(self, x):
        x_reshaped = tf.reshape(x, (-1, self.H, self.W // 2, 2, self.C))
        x1, x2 = tf.unstack(x_reshaped, axis=3)
        context = tf.concat([x1, x2], axis=3)
        context = self.processor(context)

        self.eps = tf.random.normal(x.shape)
        self.log_det_eps = tf.reduce_sum(tfd.Normal(0., 1.).log_prob(self.eps), axis=[1, 2, 3])

        return self.bijector.forward(self.eps, context=context) + x

    def _inverse(self, y):
        return self.bijector.inverse(y) + y

    def _forward_log_det_jacobian(self, x):
        x_reshaped = tf.reshape(x, (-1, self.H, self.W // 2, 2, self.C))
        x1, x2 = tf.unstack(x_reshaped, axis=3)
        context = tf.concat([x1, x2], axis=3)
        context = self.processor(context)
        log_det = self.bijector.forward_log_det_jacobian(self.eps, context=context, event_ndims=3)
        return log_det - self.log_det_eps
