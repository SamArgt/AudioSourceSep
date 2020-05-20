import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
from flow_tfp_bijectors import *
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


class RealNVPStep(tfb.Bijector):
    """
    Real NVP Step: 3 affine coupling layers with alternate masking

    Parameter:
        event_shape: list of int
        shift_and_log_scale_layer: tfk.layers.Layer class
        n_hidden_units : number of hidden units in the shift_and_log_scale layer
            if shift_anf_log_scale_layer is a ConvNet: number of filters in the hidden layers
            if it is a DenseNet: number of units in the hidden layers

    """

    def __init__(self, event_shape, shift_and_log_scale_layer,
                 n_hidden_units, masking, dtype=tf.float32):
        super(RealNVPStep, self).__init__(forward_min_event_ndims=0)

        self.coupling_layer_1 = AffineCouplingLayerMasked(
            event_shape, shift_and_log_scale_layer, n_hidden_units, masking, 0, dtype=dtype)

        self.coupling_layer_2 = AffineCouplingLayerMasked(
            event_shape, shift_and_log_scale_layer, n_hidden_units, masking, 1, dtype=dtype)

        self.coupling_layer_3 = AffineCouplingLayerMasked(
            event_shape, shift_and_log_scale_layer, n_hidden_units, masking, 0, dtype=dtype)

        self.bijector = tfb.Chain(
            [self.coupling_layer_3,
             self.coupling_layer_2,
             self.coupling_layer_1])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, x):
        return self.bijector.inverse(x)

    def _forward_log_det_jacobian(self, x):
        return self.bijector._forward_log_det_jacobian(x)


class RealNVPBlock(tfb.Bijector):
    """
    Real NVP Block :
    real nvp step (checkboard) -> squeezing -> real nvp step (channel)
    !! This bijector accepts only 3 dimensionals event_shape !!

    Parameter:
        event_shape: list of int [H, W, C]
        shift_and_log_scale_layer: layer class

    """

    def __init__(self, event_shape, shift_and_log_scale_layer,
                 n_filters, batch_norm=False):
        super(RealNVPBlock, self).__init__(forward_min_event_ndims=3)

        self.coupling_step_1 = RealNVPStep(event_shape,
                                           shift_and_log_scale_layer,
                                           n_filters, 'checkboard')
        self.squeeze = Squeeze(event_shape)
        self.event_shape_out = self.squeeze.event_shape_out
        self.coupling_step_2 = RealNVPStep(self.event_shape_out,
                                           shift_and_log_scale_layer,
                                           n_filters, 'channel')

        self.bijector = tfb.Chain(
            [self.coupling_step_2, self.squeeze, self.coupling_step_1])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, x):
        return self.bijector.inverse(x)

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


class RealNVPBijector(tfb.Bijector):

    """
    Real NVP Bijector :
    2 real NVP Blocks with a multiscale architecture as described in  the Real NVP paper
    !! This bijector accepts only 3 dimensionals event_shape !!

    Parameter:
        event_shape: list of int [H, W, C]
        shift_and_log_scale_layer: layer class
    """

    def __init__(self, input_shape, shift_and_log_scale_layer,
                 n_filters_base, batch_norm=False):
        super(RealNVPBijector, self).__init__(forward_min_event_ndims=3)

        self.real_nvp_block_1 = RealNVPBlock(input_shape,
                                             shift_and_log_scale_layer,
                                             n_filters_base, batch_norm)

        H1, W1, C1 = self.real_nvp_block_1.event_shape_out

        self.real_nvp_block_2 = RealNVPBlock([H1, W1, C1 // 2],
                                             shift_and_log_scale_layer,
                                             2 * n_filters_base, batch_norm)

    def _forward(self, x):
        output1 = self.real_nvp_block_1.forward(x)
        z1, h1 = tf.split(output1, 2, axis=-1)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H // 2, W // 2, 4 * C))
        z2 = self.real_nvp_block_2.forward(h1)
        return tf.concat((z1, z2), axis=-1)

    def _inverse(self, y):
        z1, z2 = tf.split(y, 2, axis=-1)
        h1 = self.real_nvp_block_2.inverse(z2)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H * 2, W * 2, C // 4))
        output1 = tf.concat((z1, h1), axis=-1)
        return self.real_nvp_block_1.inverse(output1)

    def _forward_log_det_jacobian(self, x):
        output1 = self.real_nvp_block_1.forward(x)
        log_det_1 = self.real_nvp_block_1.forward_log_det_jacobian(
            x, event_ndims=3)
        z1, h1 = tf.split(output1, 2, axis=-1)
        log_det_2 = self.real_nvp_block_2.forward_log_det_jacobian(
            h1, event_ndims=3)
        return log_det_1 + log_det_2

    def _forward_event_shape_tensor(self, input_shape):
        H, W, C = input_shape
        return (H // 4, W // 4, C * 16)

    def _forward_event_shape(self, input_shape):
        H, W, C = input_shape
        return (H // 4, W // 4, C * 16)

    def _inverse_event_shape_tensor(self, output_shape):
        H, W, C = output_shape
        return (H * 4, W * 4, C // 16)

    def _inverse_event_shape(self, output_shape):
        H, W, C = output_shape
        return (H * 4, W * 4, C // 16)


class RealNVPBijector2(tfb.Bijector):
    """
    Real NVP Bijector :
    2 real NVP Blocks + additionnal Real NVP Step
        with a multiscale architecture as described in  the Real NVP paper
    !! This bijector accepts only 3 dimensionals event_shape !!

    Parameter:
        event_shape: list of int [H, W, C]
        shift_and_log_scale_layer: layer class
    """

    def __init__(self, input_shape, shift_and_log_scale_layer,
                 n_filters_base, batch_norm=False):
        super(RealNVPBijector2, self).__init__(forward_min_event_ndims=3)

        self.real_nvp_block_1 = RealNVPBlock(input_shape,
                                             shift_and_log_scale_layer,
                                             n_filters_base, batch_norm)

        H1, W1, C1 = self.real_nvp_block_1.event_shape_out

        self.real_nvp_block_2 = RealNVPBlock([H1, W1, C1 // 2],
                                             shift_and_log_scale_layer,
                                             2 * n_filters_base, batch_norm)

        H2, W2, C2 = self.real_nvp_block_2.event_shape_out

        self.real_nvp_step = RealNVPStep([H2, W2, C2 // 2],
                                         shift_and_log_scale_layer,
                                         2 * n_filters_base, masking='checkboard')

    def _forward(self, x):
        output1 = self.real_nvp_block_1.forward(x)
        z1, h1 = tf.split(output1, 2, axis=-1)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H // 2, W // 2, 4 * C))
        output2 = self.real_nvp_block_2.forward(h1)
        z2, h2 = tf.split(output2, 2, axis=-1)
        z3 = self.real_nvp_step.forward(h2)
        return tf.concat((z1, z2, z3), axis=-1)

    def _inverse(self, y):
        z1, z23 = tf.split(y, 2, axis=-1)
        z2, z3 = tf.split(z23, 2, axis=-1)
        h2 = self.real_nvp_step.inverse(z3)
        output2 = tf.concat((z2, h2), axis=-1)
        h1 = self.real_nvp_block_2.inverse(output2)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H * 2, W * 2, C // 4))
        output1 = tf.concat((z1, h1), axis=-1)
        return self.real_nvp_block_1.inverse(output1)

    def _forward_log_det_jacobian(self, y):
        output1 = self.real_nvp_block_1.forward(y)
        log_det_1 = self.real_nvp_block_1.forward_log_det_jacobian(
            y, event_ndims=3)
        z1, h1 = tf.split(output1, 2, axis=-1)
        log_det_2 = self.real_nvp_block_2.forward_log_det_jacobian(
            h1, event_ndims=3)
        output2 = self.real_nvp_block_2.forward(h1)
        z2, h2 = tf.split(output2, 2, axis=-1)
        log_det_3 = self.real_nvp_step.forward_log_det_jacobian(
            h2, event_ndims=3)
        return log_det_1 + log_det_2 + log_det_3

    def _forward_event_shape_tensor(self, input_shape):
        H, W, C = input_shape
        return (H // 4, W // 4, C * 16)

    def _forward_event_shape(self, input_shape):
        H, W, C = input_shape
        return (H // 4, W // 4, C * 16)

    def _inverse_event_shape_tensor(self, output_shape):
        H, W, C = output_shape
        return (H * 4, W * 4, C // 16)

    def _inverse_event_shape(self, output_shape):
        H, W, C = output_shape
        return (H * 4, W * 4, C // 16)