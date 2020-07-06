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
        self.meanpool = tfk.layers.AveragePooling2D(pool_size=(5, 5), strides=1, padding='same')

    def call(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
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

    def call(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.norms[i * self.n_stages + j](x, y)
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
        super(CondRefineBlock, self).__init__(name=name)
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
                 dilation=None, name="ConditionalResidualBlock"):
        super(ConditionalResidualBlock, self).__init__(name=name)
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        if resample == 'down':
            if dilation is not None:
                self.conv1 = tfk.layers.Conv2D(input_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same')
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same')
                self.conv_shortcut = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                                       padding='same')
            else:
                self.conv1 = tfk.layers.Conv2D(input_dim, 3, strides=1, padding='same', use_bias=False)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = tfk.Sequential([tfk.layers.Conv2D(output_dim, 3, padding='same'),
                                             tfk.layers.AveragePooling2D(pool_size=2)])
                self.conv_shortcut = tfk.Sequential([tfk.layers.Conv2D(output_dim, 1, padding='same'),
                                                     tfk.layers.AveragePooling2D(pool_size=2)])

        elif resample is None:
            if dilation is not None:
                self.conv_shortcut = tfk.layers.Conv2D(input_dim, kernel_size=3, dilation_rate=dilation,
                                                       padding='same')
                self.conv1 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same')
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same')
            else:
                self.conv_shortcut = tfk.layers.Conv2D(output_dim, 3, strides=1, padding='same', use_bias=False)
                self.conv1 = tfk.layers.Conv2D(output_dim, 3, strides=1, padding='same', use_bias=False)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = tfk.layers.Conv2D(output_dim, 3, strides=1, padding='same', use_bias=False)
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
            shortcut = self.conv_shortcut(x)

        return shortcut + output


class ConditionalInstanceNorm2dPlus(tfk.layers.Layer):
    def __init__(self, num_features, num_classes, bias=True, name="ConditionalInstanceNorm2dPlus"):
        super(ConditionalInstanceNorm2dPlus, self).__init__(name=name)
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
        means = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        m, v = tf.nn.moments(means, axes=-1, keepdims=True)
        means = (means - m) / tf.math.sqrt(v + 1e-5)
        h = self.instance_norm(x)
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
                                            input_shape=input_shape)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = tfk.layers.Conv2D(input_shape[-1], 3, strides=1, padding='same')

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

    def sample(self, n_samples, sigmas, n_steps_each=100, step_lr=0.00002, return_arr=False):
        """
        Anneal Langevin dynamics
        """
        x_mod = tf.random.uniform([n_samples] + list(self.data_shape))
        if return_arr:
            x_arr = [x_mod]
        for i, sigma in enumerate(sigmas):
            labels = tf.expand_dims(tf.ones(n_samples) * i, -1)
            step_size = tf.constant(step_lr * (sigma / sigmas[-1]) ** 2, dtype=tf.float32)
            for s in range(n_steps_each):
                noise = tf.random.normal((n_samples,)) * tf.math.sqrt(step_size * 2)
                grad = self.call(x_mod, labels)
                x_mod = x_mod + step_size * grad + tf.reshape(noise, (n_samples, 1, 1, 1))
                if return_arr:
                    x_arr.append(tf.clip_by_value(x_mod, 0., 1.))

        if return_arr:
            return x_arr
        else:
            return tf.clip_by_value(x_mod, 0., 1.)
