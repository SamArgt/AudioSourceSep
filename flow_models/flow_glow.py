import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
from .flow_tfp_bijectors import *
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


class GlowStep(tfb.Bijector):

    def __init__(self, event_shape, shift_and_log_scale_layer, n_hidden_units, minibatch, name='glowStep', **kwargs):

        super(GlowStep, self).__init__(forward_min_event_ndims=3)

        self.actnorm = ActNorm(event_shape, minibatch, name=name + '/ActNorm')
        self.inv1x1conv = Invertible1x1Conv(
            event_shape, name=name + '/inv1x1conv')
        # self.coupling_layer = AffineCouplingLayerMasked(event_shape, shift_and_log_scale_layer,
        #                                                n_hidden_units, name=name + '/couplingLayer')
        self.coupling_layer = AffineCouplingLayerSplit(event_shape, shift_and_log_scale_layer,
                                                       n_hidden_units, name=name + '/couplingLayer', **kwargs)
        self.bijector = tfb.Chain(
            [self.coupling_layer, self.inv1x1conv, self.actnorm])

    def _forward(self, x):
        return self.bijector.forward(x)

    def _inverse(self, y):
        return self.bijector.inverse(y)

    def _forward_log_det_jacobian(self, x):
        return self.bijector.forward_log_det_jacobian(x, event_ndims=3)


class GlowBlock(tfb.Bijector):

    def __init__(self, K, event_shape, shift_and_log_scale_layer, n_hidden_units, minibatch, name='glowBlock', **kwargs):

        super(GlowBlock, self).__init__(forward_min_event_ndims=3)

        self.squeeze = Squeeze(event_shape)
        self.event_shape_out = self.squeeze.event_shape_out
        minibatch_updated = self.squeeze.forward(minibatch)
        self.glow_steps = []
        for k in range(K):
            glow_step = GlowStep(self.event_shape_out,
                                 shift_and_log_scale_layer,
                                 n_hidden_units, minibatch_updated,
                                 name=name + '/' + 'glowStep' + str(k), **kwargs)
            minibatch_updated = glow_step.forward(minibatch_updated)
            self.glow_steps.append(glow_step)

        self.chain = self.glow_steps + [self.squeeze]
        self.bijector = tfb.Chain(self.chain)

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


class GlowBijector_2blocks(tfb.Bijector):

    def __init__(self, K, event_shape, shift_and_log_scale_layer, n_hidden_units, minibatch, **kwargs):

        super(GlowBijector_2blocks, self).__init__(forward_min_event_ndims=3)

        H, W, C = event_shape

        self.glow_block1 = GlowBlock(K, event_shape,
                                     shift_and_log_scale_layer,
                                     n_hidden_units, minibatch,
                                     name='glowBlock1', **kwargs)

        H1, W1, C1 = self.glow_block1.event_shape_out
        minibatch_updated = self.glow_block1.forward(minibatch)
        _, minibatch_updated = tf.split(minibatch_updated, 2, axis=-1)

        self.glow_block2 = GlowBlock(K, [H1, W1, C1 // 2],
                                     shift_and_log_scale_layer,
                                     n_hidden_units, minibatch_updated,
                                     name='glowBlock2', **kwargs)

    def _forward(self, x):
        output1 = self.glow_block1.forward(x)
        z1, h1 = tf.split(output1, 2, axis=-1)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H // 2, W // 2, 4 * C))
        z2 = self.glow_block2.forward(h1)
        return tf.concat((z1, z2), axis=-1)

    def _inverse(self, y):
        z1, z2 = tf.split(y, 2, axis=-1)
        h1 = self.glow_block2.inverse(z2)
        N, H, W, C = z1.shape
        z1 = tf.reshape(z1, (N, H * 2, W * 2, C // 4))
        output1 = tf.concat((z1, h1), axis=-1)
        x = self.glow_block1.inverse(output1)
        return x

    def _forward_log_det_jacobian(self, x):
        output1 = self.glow_block1.forward(x)
        log_det_1 = self.glow_block1.forward_log_det_jacobian(
            x, event_ndims=3)
        z1, h1 = tf.split(output1, 2, axis=-1)
        log_det_2 = self.glow_block2.forward_log_det_jacobian(
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


class GlowBijector_3blocks(tfb.Bijector):

    def __init__(self, K, event_shape, shift_and_log_scale_layer, n_hidden_units, minibatch, **kwargs):

        super(GlowBijector_3blocks, self).__init__(forward_min_event_ndims=3)

        self.H, self.W, self.C = event_shape

        self.glow_block1 = GlowBlock(K, event_shape,
                                     shift_and_log_scale_layer,
                                     n_hidden_units, minibatch,
                                     name='glowBlock1', **kwargs)

        H1, W1, C1 = self.glow_block1.event_shape_out
        minibatch_updated = self.glow_block1.forward(minibatch)
        _, minibatch_updated = tf.split(minibatch_updated, 2, axis=-1)

        self.glow_block2 = GlowBlock(K, [H1, W1, C1 // 2],
                                     shift_and_log_scale_layer,
                                     n_hidden_units, minibatch_updated,
                                     name='glowBlock2', **kwargs)

        H2, W2, C2 = self.glow_block2.event_shape_out
        minibatch_updated = self.glow_block2.forward(minibatch_updated)
        _, minibatch_updated = tf.split(minibatch_updated, 2, axis=-1)

        self.glow_block3 = GlowBlock(K, [H2, W2, C2 // 2],
                                     shift_and_log_scale_layer,
                                     n_hidden_units, minibatch_updated,
                                     name='glowBlock3', **kwargs)

    def _forward(self, x):
        output1 = self.glow_block1.forward(x)
        z1, h1 = tf.split(output1, 2, axis=-1)
        z1 = tf.reshape(z1, (-1, self.H // 8, self.W // 8, 32 * self.C))
        output2 = self.glow_block2.forward(h1)
        z2, h2 = tf.split(output2, 2, axis=-1)
        z2 = tf.reshape(z2, (-1, self.H // 8, self.W // 8, 16 * self.C))
        z3 = self.glow_block3.forward(h2)
        return tf.concat((z1, z2, z3), axis=-1)

    def _inverse(self, y):
        z1, z2, z3 = tf.split(y, 3, axis=-1)
        h2 = self.glow_block3.inverse(z3)
        z2 = tf.reshape(z2, (-1, self.H * 8, self.W * 8, self.C // 16))
        output2 = tf.concat((z2, h2), axis=-1)
        h1 = self.glow_block2.inverse(output2)
        z1 = tf.reshape(z1, (-1, self.H * 8, self.W * 8, self.C // 32))
        output1 = tf.concat((z1, h1), axis=-1)
        return self.glow_block1.inverse(output1)

    def _forward_log_det_jacobian(self, x):
        output1 = self.glow_block1.forward(x)
        log_det_1 = self.glow_block1.forward_log_det_jacobian(
            x, event_ndims=3)
        z1, h1 = tf.split(output1, 2, axis=-1)
        output2 = self.glow_block2.forward(h1)
        log_det_2 = self.glow_block2.forward_log_det_jacobian(
            h1, event_ndims=3)
        z2, h2 = tf.split(output2, 2, axis=-1)
        log_det_3 = self.glow_block3._forward_log_det_jacobian(
            h2, event_ndims=3)
        return log_det_1 + log_det_2 + log_det_3

    def _forward_event_shape_tensor(self, input_shape):
        H, W, C = input_shape
        return (H // 8, W // 8, C * 64)

    def _forward_event_shape(self, input_shape):
        H, W, C = input_shape
        return (H // 8, W // 8, C * 64)

    def _inverse_event_shape_tensor(self, output_shape):
        H, W, C = output_shape
        return (H * 8, W * 8, C // 64)

    def _inverse_event_shape(self, output_shape):
        H, W, C = output_shape
        return (H * 8, W * 8, C // 64)
