import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
tfk = tf.keras


class CondCRPBlock(tfk.layers.Layer):
    def __init__(self, features, n_stages, num_classes, normalizer, act=tf.nn.relu, name='CondCRPBlock'):
        super(CondCRPBlock, self).__init__(name=name)
        self.convs = []
        self.norms = []
        for i in range(n_stages):
            self.convs.append(tfk.layers.Conv2D(features, kernel_size=3, strides=1, use_bias=False, padding='same',
                                                name=name + '/conv_{}'.format(i + 1)))
            self.norms.append(normalizer(features, num_classes, bias=True, name=name + '/norm_{}'.format(i + 1)))
        self.n_stages = n_stages
        self.act = tf.keras.layers.Activation(act)
        self.meanpool = tfk.layers.AveragePooling2D(pool_size=(5, 5), strides=1, padding='same', name=name + "/AveragePooling2D")

    def call(self, x, y, training=False):
        x = self.act(x)
        path = tf.identity(x)
        for i in range(self.n_stages):
            path = self.norms[i](path, y, training=training)
            path = self.meanpool(path)
            path = self.convs[i](path)
            x += path
        return x


class CondRCUBlock(tfk.layers.Layer):
    def __init__(self, features, n_blocks, n_stages, num_classes, normalizer, act=tf.nn.relu, name="CondRCUBlock"):
        super(CondRCUBlock, self).__init__(name=name)
        self.convs = []
        self.norms = []
        for i in range(n_blocks):
            for j in range(n_stages):
                self.convs.append(tfk.layers.Conv2D(features, kernel_size=3, strides=1, use_bias=False, padding='same',
                                                    name=name + '/conv_{}_{}'.format(i + 1, j + 1)))
                self.norms.append(normalizer(features, num_classes, bias=True,
                                             name=name + '/norm_{}_{}'.format(i + 1, j + 1)))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def call(self, x, y, training=False):
        for i in range(self.n_blocks):
            residual = tf.identity(x)
            for j in range(self.n_stages):
                x = self.norms[i * self.n_stages + j](x, y, training=training)
                x = self.convs[i * self.n_stages + j](x)
            x += residual
        return x


class CondMSFBlock(tfk.layers.Layer):
    def __init__(self, in_planes, features, num_classes, normalizer, name="CondMSFBlock"):
        super(CondMSFBlock, self).__init__(name=name)
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.features = features
        self.convs = []
        self.norms = []
        for i in range(len(in_planes)):
            self.convs.append(tfk.layers.Conv2D(features, kernel_size=3, strides=1, use_bias=True, padding='same',
                                                name=name + '/conv_{}'.format(i + 1)))
            self.norms.append(normalizer(in_planes[i], num_classes, bias=True, name=name + '/norm_{}'.format(i + 1)))

    def call(self, xs, y, shape, training=False):
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y, training=training)
            h = self.convs[i](h)
            h = tf.image.resize(h, size=shape)
            if i == 0:
                sums = tf.identity(h)
            else:
                sums += h
        return sums


class CondRefineBlock(tfk.layers.Layer):
    def __init__(self, in_planes, features, num_classes, normalizer,
                 act=tf.nn.relu, start=False, end=False, name="CondRefineBlock"):
        super(CondRefineBlock, self).__init__(name=name)
        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = []
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act,
                             name=name + '/CondRCUBlock_{}'.format(i + 1))
            )

        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act, name=name + "/CondRCUBlock_output")

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer, name=name + '/CondMSFBlock')

        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act, name=name + "/CondCRPBlock")

    def call(self, xs, y, output_shape, training=False):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y, training=training)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape, training=training)
        else:
            h = hs[0]

        h = self.crp(h, y, training=training)
        h = self.output_convs(h, y, training=training)

        return h


class ConditionalResidualBlock(tfk.layers.Layer):
    def __init__(self, input_dim, output_dim, num_classes, normalization, resample=None, act=tf.nn.elu,
                 dilation=None, name="ConditionalResidualBlock"):
        super(ConditionalResidualBlock, self).__init__(name=name)
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        if resample == 'down':
            if dilation is not None:
                self.conv1 = tfk.layers.Conv2D(input_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same', name=name + '/conv1')
                self.normalize2 = normalization(input_dim, num_classes, name=name + '/norm2')
                self.conv2 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same', name=name + "/conv2")
                self.conv_shortcut = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                                       padding='same', name=name + '/shortcut')
            else:
                self.conv1 = tfk.layers.Conv2D(input_dim, 3, strides=1, padding='same', use_bias=False, name=name + '/conv1')
                self.normalize2 = normalization(input_dim, num_classes, name=name + '/norm2')
                self.conv2 = tfk.Sequential([tfk.layers.Conv2D(output_dim, 3, padding='same'),
                                             tfk.layers.AveragePooling2D(pool_size=2)], name=name + "/conv2")
                self.conv_shortcut = tfk.Sequential([tfk.layers.Conv2D(output_dim, 1, padding='same'),
                                                     tfk.layers.AveragePooling2D(pool_size=2)], name=name + "/shortcut")

        elif resample is None:
            if dilation is not None:
                self.conv_shortcut = tfk.layers.Conv2D(input_dim, kernel_size=3, dilation_rate=dilation,
                                                       padding='same', name=name + "shortcut")
                self.conv1 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same', name=name + '/conv1')
                self.normalize2 = normalization(output_dim, num_classes, name=name + '/norm2')
                self.conv2 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same', name=name + "/conv2")
            else:
                self.conv_shortcut = tfk.layers.Conv2D(output_dim, 3, strides=1, padding='same', use_bias=False, name=name + "/shortcut")
                self.conv1 = tfk.layers.Conv2D(output_dim, 3, strides=1, padding='same', use_bias=False, name=name + "/conv1")
                self.normalize2 = normalization(output_dim, num_classes, name=name + "/norm2")
                self.conv2 = tfk.layers.Conv2D(output_dim, 3, strides=1, padding='same', use_bias=False, name=name + "/conv2")
        else:
            raise Exception('invalid resample value')

        self.normalize1 = normalization(input_dim, num_classes, name=name + "/norm1")

    def call(self, x, y, training=False):
        output = self.normalize1(x, y, training=training)
        output = self.act(output)
        output = self.conv1(output)
        output = self.normalize2(output, y, training=training)
        output = self.act(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = tf.identity(x)
        else:
            shortcut = self.conv_shortcut(x)

        return shortcut + output


class ConditionalInstanceNorm2dPlus(tfk.layers.Layer):
    def __init__(self, num_features, num_classes, bias=True, name="ConditionalInstanceNorm2dPlus"):
        super(ConditionalInstanceNorm2dPlus, self).__init__(name=name)
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = tfa.layers.InstanceNormalization(name=name + '/instance_norm')
        self.gamma_embed = tfk.layers.Embedding(num_classes, num_features,
                                                embeddings_initializer=tf.random_normal_initializer(mean=0.0,
                                                                                                    stddev=0.02),
                                                name=name + '/gamma_embed')
        self.alpha_embed = tfk.layers.Embedding(num_classes, num_features,
                                                embeddings_initializer=tf.random_normal_initializer(mean=0.0,
                                                                                                    stddev=0.02),
                                                name=name + '/alpha_embed')
        if bias:
            self.beta_embed = tfk.layers.Embedding(num_classes, num_features,
                                                   embeddings_initializer="zeros",
                                                   name=name + '/beta_embed')

    def call(self, x, y, training=False):
        means = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        m, v = tf.nn.moments(means, axes=-1, keepdims=True)
        means = (means - m) / tf.math.sqrt(v + 1e-5)
        h = self.instance_norm(x, training=training)
        gamma = self.gamma_embed(y)
        gamma = tf.reshape(gamma, (-1, 1, 1, self.num_features))
        alpha = self.alpha_embed(y)
        alpha = tf.reshape(alpha, (-1, 1, 1, self.num_features))
        if self.bias:
            beta = self.beta_embed(y)
            beta = tf.reshape(beta, (-1, 1, 1, self.num_features))
        else:
            beta = 0.

        out = gamma * h + means * alpha + beta
        return out


class CondRefineNetDilated(tfk.layers.Layer):
    def __init__(self, input_shape, ngf, num_classes, logit_transform=False):
        super(CondRefineNetDilated, self).__init__()
        self.logit_transform = logit_transform
        self.norm = ConditionalInstanceNorm2dPlus
        self.ngf = ngf
        self.num_classes = num_classes
        self.act = act = tf.nn.elu
        self.data_shape = input_shape

        self.begin_conv = tfk.layers.Conv2D(ngf, 3, strides=1, padding='same',
                                            input_shape=input_shape, name="begin_conv")
        self.normalizer = self.norm(ngf, self.num_classes, name='normalizer')

        self.end_conv = tfk.layers.Conv2D(input_shape[-1], 3, strides=1, padding='same', name='end_conv')

        self.res1 = [
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, name="Res1_1"),
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, name="Res1_2")]

        self.res2 = [
            ConditionalResidualBlock(self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, name="Res2_1"),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, name="Res2_2")]

        self.res3 = [
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=2, name="Res3_1"),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=2, name="Res3_2")]

        self.res4 = [
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=4, name="Res4_1"),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=4, name="Res4_2")]

        self.refine1 = CondRefineBlock([2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act, start=True, name="refine1")
        self.refine2 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act, name="refine2")
        self.refine3 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, self.num_classes, self.norm, act=act, name="refine3")
        self.refine4 = CondRefineBlock([self.ngf, self.ngf], self.ngf, self.num_classes, self.norm, act=act, end=True, name="refine4")

    def _compute_cond_module(self, module, x, y, training=False):
        for m in module:
            x = m(x, y, training=training)
        return x

    def call(self, x, y, training=False):
        if not self.logit_transform:
            x = 2 * x - 1.

        output = self.begin_conv(x)

        layer1 = self._compute_cond_module(self.res1, output, y, training=training)
        layer2 = self._compute_cond_module(self.res2, layer1, y, training=training)
        layer3 = self._compute_cond_module(self.res3, layer2, y, training=training)
        layer4 = self._compute_cond_module(self.res4, layer3, y, training=training)

        ref1 = self.refine1([layer4], y, layer4.shape[1:3], training=training)
        ref2 = self.refine2([layer3, ref1], y, layer3.shape[1:3], training=training)
        ref3 = self.refine3([layer2, ref2], y, layer2.shape[1:3], training=training)
        output = self.refine4([layer1, ref3], y, layer1.shape[1:3], training=training)

        output = self.normalizer(output, y, training=training)
        output = self.act(output)
        output = self.end_conv(output)
        return output
