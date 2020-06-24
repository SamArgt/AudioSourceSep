import tensorflow as tf
import tensorflow_probability as tfp
from .flow_tfp_bijectors import *
from .flow_tfk_layers import *
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


class FlowppCouplingLayer(tfp.bijectors.Bijector):

    def __init__(self, input_shape, minibatch, split='channel', split_state=0, n_components=32, n_blocks=10, filters=96,
                 dropout_p=0., heads=4, name="FlowppCouplingLayer"):

        super(FlowppCouplingLayer, self).__init__(forward_min_event_ndims=3)

        self.actnorm = ActNorm(input_shape, minibatch,
                               normalize='all', name=name + "/actnorm")
        self.inv1x1conv = Invertible1x1Conv(
            input_shape, name=name + '/inv1x1conv')
        NN = ConvAttnNet
        self.mixLogCdfAttnCoupling = MixLogisticCDFAttnCoupling(input_shape, NN, split=split, split_state=split_state,
                                                                n_components=n_components, n_blocks=n_blocks, filters=filters,
                                                                dropout_p=dropout_p, heads=heads, name=name + "/mixLogCdfAttnCoupling")

        self.bijector = tfb.Chain(
            [self.mixLogCdfAttnCoupling, self.inv1x1conv, self.actnorm])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, y):
        return self.bijector.inverse(y)

    def _forward_log_det_jacobian(self, x):
        return self.bijector.forward_log_det_jacobian(x, event_ndims=3)


class FlowppBlock(tfp.bijectors.Bijector):

    def __init__(self, input_shape, minibatch, n_layers, split="channel", n_components=32, n_blocks=10, filters=96,
                 dropout_p=0., heads=4, name="FlowppBlock"):

        super(FlowppBlock, self).__init__(forward_min_event_ndims=3)

        self.coupling_layers = []
        split_state = 0
        minibatch_updated = minibatch
        for i in range(n_layers, 0, -1):

            coupling_layer = FlowppCouplingLayer(input_shape, minibatch_updated, split=split,
                                                 split_state=split_state, n_components=n_components,
                                                 n_blocks=n_blocks, filters=filters,
                                                 dropout_p=dropout_p, heads=heads,
                                                 name=name + "/FlowppCouplingLayer" + str(i))
            minibatch_updated = coupling_layer.forward(minibatch_updated)
            self.coupling_layers.append(coupling_layer)
            split_state = split_state + 1 % 2

        self.bijector = tfb.Chain(self.coupling_layers)

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, y):
        return self.bijector.inverse(y)

    def _forward_log_det_jacobian(self, x):
        return self.bijector.forward_log_det_jacobian(x, event_ndims=3)


class Flowpp_cifar10(tfp.bijectors.Bijector):

    def __init__(self, input_shape, minibatch, n_components=32, n_blocks=10, filters=96,
                 dropout_p=0., heads=4, name="flowpp_cifar10"):

        super(Flowpp_cifar10, self).__init__(forward_min_event_ndims=3)

        self.flow_block1 = FlowppBlock(input_shape, minibatch, 4, split="checkerboard", n_components=n_components,
                                       n_blocks=n_blocks, filters=filters,
                                       dropout_p=dropout_p, heads=heads,
                                       name=name + "/flowBlock1")
        minibatch_updated = self.flow_block1(minibatch)

        self.squeeze = Squeeze(input_shape)
        minibatch_updated = self.squeeze(minibatch_updated)
        event_shape_out = self.squeeze.event_shape_out

        self.flow_block2 = FlowppBlock(event_shape_out, minibatch_updated, 2, split="channel",
                                       n_components=n_components,
                                       n_blocks=n_blocks, filters=filters,
                                       dropout_p=dropout_p, heads=heads,
                                       name=name + "/flowBlock2")

        minibatch_updated = self.flow_block2(minibatch_updated)

        self.flow_block3 = FlowppBlock(event_shape_out, minibatch_updated,
                                       3, split="checkerboard", n_components=n_components,
                                       n_blocks=n_blocks, filters=filters,
                                       dropout_p=dropout_p, heads=heads,
                                       name=name + "/flowBlock3")

        self.bijector = tfb.Chain(
            [self.flow_block3, self.flow_block2, self.squeeze, self.flow_block1])

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

    def __init__(self, input_shape, minibatch, n_components=32, n_blocks=2, filters=96,
                 dropout_p=0., heads=4, name="dequant_flowpp"):

        super(DequantFlowpp, self).__init__(forward_min_event_ndims=3)

        self.bijector = FlowppBlock(input_shape, minibatch, 4, split="checkerboard",
                                    n_components=n_components, n_blocks=n_blocks, filters=filters,
                                    dropout_p=dropout_p, heads=heads, name=name + "flow_block")

    def _forward(self, x):
        return self.bijector.forward(x) + x

    def _inverse(self, y):
        return self.bijector.inverse(y) + y

    def _forward_log_det_jacobian(self, x):
        return self.bijector.forward_log_det_jacobian(x, event_ndims=3)
