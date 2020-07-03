import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
tfk = tf.keras


class CondCRPBlock(tfk.layers.Layer):
    def __init__(self, features, n_stages, num_classes, normalizer, act=tf.nn.relu, name='CondCRPBlock'):
        super(CondCRPBlock, self).__init__()
        self.convs = []
        self.norms = []
        for i in range(n_stages):
            self.convs.append(tfk.layers.Conv2D(features, kernel_size=3, strides=1, use_bias=False, padding='same',
                                                name=name + '/conv_{}'.format(i + 1)))
            self.norms.append(normalizer(features, num_classes, bias=True, name=name + '/norm_{}'.format(i + 1)))
        self.n_stages = n_stages
        self.act = act
        self.meanpool = tfk.layers.AveragePooling2D(pool_size=(5, 5), strides=1, padding='same')

    def call(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.maxpool(path)
            path = self.convs[i](path)
            x += path
        return x


class CondRCUBlock(tfk.layers.Layer):
    def __init__(self, features, n_blocks, n_stages, num_classes, normalizer, act=tf.nn.relu, name="CondRCUBlock"):
        super(CondRCUBlock, self).__init__()
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

    def call(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.norms[i * self.n_blocks + j](x, y)
                x = self.convs[i * self.n_blocks + j](x)
            x += residual


class CondMSFBlock(tfk.layers.Layer):
    def __init__(self, in_planes, features, num_classes, normalizer, name="CondMSFBlock"):
        super(CondMSFBlock, self).__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.features = features
        self.convs = []
        self.norms = []
        for i in range(len(in_planes)):
            self.convs.append(tfk.layers.Conv2D(features, kernel_size=3, strides=1, use_bias=True, padding='same',
                                                name=name + '/conv_{}'.format(i + 1)))
            self.norms.append(normalizer(features, num_classes, bias=True, name=name + '/norm_{}'.format(i + 1)))

    def call(self, xs, y, shape):
        sums = tf.zeros(shape=(xs[0].shape[0], shape[0], shape[1], self.features))
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = tf.image.resize(h, size=shape)
            sums += h
        return sums


class CondRefineBlock(tfk.layers.Layer):
    def __init__(self, in_planes, features, num_classes, normalizer,
                 act=tf.nn.relu, start=False, end=False, name="CondRefineBlock"):
        super(CondRefineBlock, self).__init__()
        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = []
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act,
                             name=name + '/CondRCUBlock_{}'.format(i + 1))
            )

        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act)

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer)

        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act)

    def call(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]

        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h


class ConditionalResidualBlock(tfk.layers.Layer):
    def __init__(self, input_dim, output_dim, num_classes, normalization, resample=None, act=tf.nn.elu,
                 adjust_padding=False, dilation=None):
        super(ConditionalResidualBlock, self).__init__()
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        if resample == 'down':
            if dilation is not None:
                self.conv1 = tfk.layers.Conv2D(input_dim, kernel_size=3, dilatation_rate=dilation,
                                               padding='same')
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilatation_rate=dilation,
                                               padding='same')
                self.conv_shortcut = tfk.layers.Conv2D(output_dim, kernel_size=3, dilatation_rate=dilation,
                                                       padding='same')
            else:
                self.conv1 = tfk.layers.Conv2d(input_dim, 3, stride=1, padding='same', use_bias=False)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = tfk.Sequential([tfk.layers.Conv2D(output_dim, 3, padding='same'),
                                             tfk.layers.AveragePooling2D()])
                self.conv_shortcut = tfk.Sequential([tfk.layers.Conv2D(output_dim, 3, padding='same'),
                                                     tfk.layers.AveragePooling2D()])

        elif resample is None:
            if dilation is not None:
                self.conv_shortcut = tfk.layers.Conv2D(input_dim, kernel_size=3, dilatation_rate=dilation,
                                                       padding='same')
                self.conv1 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilatation_rate=dilation,
                                               padding='same')
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilatation_rate=dilation,
                                               padding='same')
            else:
                self.conv_shortcut = tfk.layers.Conv2d(output_dim, 3, stride=1, padding='same', use_bias=False)
                self.conv1 = tfk.layers.Conv2d(output_dim, 3, stride=1, padding='same', use_bias=False)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = tfk.layers.Conv2d(output_dim, 3, stride=1, padding='same', use_bias=False)
        else:
            raise Exception('invalid resample value')

        self.normalize1 = normalization(input_dim, num_classes)

    def call(self, x, y):
        output = self.normalize1(x, y)
        output = self.act(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.act(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class ConditionalInstanceNorm2dPlus(tfk.layers.Layer):
    def __init__(self, num_features, num_classes, bias=True):
        super(ConditionalInstanceNorm2dPlus, self).__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = tfa.layers.InstanceNormalization()
        self.gamma_embed = tfk.layers.Embedding(num_classes, num_features,
                                                embeddings_initializer=tf.random_normal_initializer(mean=0.0,
                                                                                                    stddev=0.02))
        self.alpha_embed = tfk.layers.Embedding(num_classes, num_features,
                                                embeddings_initializer=tf.random_normal_initializer(mean=0.0,
                                                                                                    stddev=0.02))
        if bias:
            self.beta_embed = tfk.layers.Embedding(num_classes, num_features,
                                                   embeddings_initializer="zeros")

    def call(self, x, y):
        means = tf.reduce_mean(x, axis=[1, 2])
        m, v = tf.nn.moments(x, axis=-1, keepdims=True)
        means = (means - m) / tf.math.sqrt(v + 1e-5)
        h = self.instance_norm(x)
        gamma = self.gamma_embed(y)
        alpha = self.alpha_embed(y)
        if self.bias:
            beta = self.beta_embed(y)
        else:
            beta = 0.
        out = gamma * h + means * alpha + beta
        return out


class CondRefineNetDilated(tfk.layers.Layer):
    def __init__(self, config):
        super(CondRefineNetDilated, self).__init__()
        self.logit_transform = config.data.logit_transform
        self.norm = ConditionalInstanceNorm2dPlus
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = tf.nn.elu

        self.begin_conv = tfk.layers.Conv2d(ngf, 3, stride=1, padding='same', input_shape=config.input_shape)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = tfk.layers.Conv2d(config.data.channels, 3, stride=1, padding='same')

        self.res1 = [
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]

        self.res2 = [
            ConditionalResidualBlock(self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]

        self.res3 = [
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=2),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=2)]

        if config.data.image_size == 28:
            self.res4 = [
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=True, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]
        else:
            self.res4 = [
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=False, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]

        self.refine1 = CondRefineBlock([2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act, start=True)
        self.refine2 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act)
        self.refine3 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, self.num_classes, self.norm, act=act)
        self.refine4 = CondRefineBlock([self.ngf, self.ngf], self.ngf, self.num_classes, self.norm, act=act, end=True)

    def _compute_cond_module(self, module, x, y):
        for m in module:
            x = m(x, y)
        return x

    def call(self, x, y):
        if not self.logit_transform:
            x = 2 * x - 1.

        output = self.begin_conv(x)

        layer1 = self._compute_cond_module(self.res1, output, y)
        layer2 = self._compute_cond_module(self.res2, layer1, y)
        layer3 = self._compute_cond_module(self.res3, layer2, y)
        layer4 = self._compute_cond_module(self.res4, layer3, y)

        ref1 = self.refine1([layer4], y, layer4.shape[1:3])
        ref2 = self.refine2([layer3, ref1], y, layer3.shape[1:3])
        ref3 = self.refine3([layer2, ref2], y, layer2.shape[1:3])
        output = self.refine4([layer1, ref3], y, layer1.shape[1:3])

        output = self.normalizer(output, y)
        output = self.act(output)
        output = self.end_conv(output)
        return output
