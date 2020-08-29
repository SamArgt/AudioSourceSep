import tensorflow as tf
import tensorflow_addons as tfa
tfk = tf.keras


class CRPBlock(tfk.layers.Layer):
    def __init__(self, features, n_stages, act=tf.nn.elu, name='CRPBlock'):
        super(CRPBlock, self).__init__(name=name)
        self.convs = []
        for i in range(n_stages):
            self.convs.append(tfk.layers.Conv2D(features, kernel_size=3, strides=1, use_bias=False, padding='same',
                                                name='conv_{}'.format(i + 1)))
        self.n_stages = n_stages
        self.act = tf.keras.layers.Activation(act)
        self.maxpool = tfk.layers.MaxPooling2D(pool_size=(
            5, 5), strides=1, padding='same', name="MaxPooling2D")

    def call(self, x):
        x = self.act(x)
        path = tf.identity(x)
        for i in range(self.n_stages):
            path = self.maxpool(path)
            path = self.convs[i](path)
            x += path
        return x


class RCUBlock(tfk.layers.Layer):
    def __init__(self, features, n_blocks, n_stages, act=tf.nn.elu, name="RCUBlock"):
        super(RCUBlock, self).__init__(name=name)
        self.convs = []
        for i in range(n_blocks):
            for j in range(n_stages):
                self.convs.append(tfk.layers.Conv2D(features, kernel_size=3, strides=1, use_bias=False, padding='same',
                                                    name='conv_{}_{}'.format(i + 1, j + 1)))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def call(self, x):
        for i in range(self.n_blocks):
            residual = tf.identity(x)
            for j in range(self.n_stages):
                x = self.convs[i * self.n_stages + j](x)
            x += residual
        return x


class MSFBlock(tfk.layers.Layer):
    def __init__(self, in_planes, features, name="MSFBlock"):
        super(MSFBlock, self).__init__(name=name)
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.features = features
        self.convs = []
        for i in range(len(in_planes)):
            self.convs.append(tfk.layers.Conv2D(features, kernel_size=3, strides=1, use_bias=True, padding='same',
                                                name='conv_{}'.format(i + 1)))

    def call(self, xs, shape):
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = tf.image.resize(h, size=shape)
            if i == 0:
                sums = tf.identity(h)
            else:
                sums += h
        return sums


class RefineBlock(tfk.layers.Layer):
    def __init__(self, in_planes, features, act=tf.nn.elu, start=False, end=False, name="RefineBlock"):
        super(RefineBlock, self).__init__(name=name)
        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = []
        for i in range(n_blocks):
            self.adapt_convs.append(
                RCUBlock(in_planes[i], 2, 2, act, name='RCUBlock_{}'.format(i + 1))
            )

        self.output_convs = RCUBlock(
            features, 3 if end else 1, 2, act, name="RCUBlock_output")

        if not start:
            self.msf = MSFBlock(in_planes, features, name='MSFBlock')

        self.crp = CRPBlock(features, 2, act, name="CRPBlock")

    def call(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)

        return h


class ResidualBlock(tfk.layers.Layer):
    def __init__(self, input_dim, output_dim, normalization, resample=None, act=tf.nn.elu,
                 dilation=None, name="ResidualBlock"):
        super(ResidualBlock, self).__init__(name=name)
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        if resample == 'down':
            if dilation is not None:
                self.conv1 = tfk.layers.Conv2D(input_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same', name="conv1")
                self.normalize2 = normalization(input_dim, name="norm2")
                self.conv2 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same', name="conv2")
                self.conv_shortcut = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                                       padding='same', name="shortcut")
            else:
                self.conv1 = tfk.layers.Conv2D(
                    input_dim, 3, strides=1, padding='same', use_bias=False, name="conv1")
                self.normalize2 = normalization(input_dim, name="norm2")
                self.conv2 = tfk.Sequential([tfk.layers.Conv2D(output_dim, 3, padding='same'),
                                             tfk.layers.AveragePooling2D(pool_size=2)], name="conv2")
                self.conv_shortcut = tfk.Sequential([tfk.layers.Conv2D(output_dim, 1, padding='same'),
                                                     tfk.layers.AveragePooling2D(pool_size=2)], name="shortcut")

        elif resample is None:
            if dilation is not None:
                self.conv_shortcut = tfk.layers.Conv2D(input_dim, kernel_size=3, dilation_rate=dilation,
                                                       padding='same', name="shortcut")
                self.conv1 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same', name="conv1")
                self.normalize2 = normalization(output_dim, name="norm2")
                self.conv2 = tfk.layers.Conv2D(output_dim, kernel_size=3, dilation_rate=dilation,
                                               padding='same', name="conv2")
            else:
                self.conv_shortcut = tfk.layers.Conv2D(
                    output_dim, 3, strides=1, padding='same', use_bias=False, name="shortcut")
                self.conv1 = tfk.layers.Conv2D(
                    output_dim, 3, strides=1, padding='same', use_bias=False, name="conv1")
                self.normalize2 = normalization(output_dim, name="norm2")
                self.conv2 = tfk.layers.Conv2D(
                    output_dim, 3, strides=1, padding='same', use_bias=False, name="conv2")
        else:
            raise Exception('invalid resample value')

        self.normalize1 = normalization(input_dim, name="norm1")

    def call(self, x, training=True):
        output = self.normalize1(x, training=training)
        output = self.act(output)
        output = self.conv1(output)
        output = self.normalize2(output, training=training)
        output = self.act(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = tf.identity(x)
        else:
            shortcut = self.conv_shortcut(x)

        return shortcut + output


class InstanceNorm2dPlus(tfk.layers.Layer):
    def __init__(self, num_features, bias=True, name="InstanceNorm2dPlus"):
        super(InstanceNorm2dPlus, self).__init__(name=name)
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = tfa.layers.InstanceNormalization()
        self.alpha = tf.Variable(initial_value=tf.random.normal(
            (num_features,), mean=0., stddev=0.02), dtype=tf.float32)
        self.gamma = tf.Variable(initial_value=tf.random.normal(
            (num_features,), mean=0., stddev=0.02), dtype=tf.float32)
        if bias:
            self.beta = tf.Variable(initial_value=tf.zeros(
                (num_features,)), dtype=tf.float32, trainable=bias)

    def call(self, x, training=True):
        means = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        m, v = tf.nn.moments(means, axes=-1, keepdims=True)
        means = (means - m) / tf.math.sqrt(v + 1e-5)
        h = self.instance_norm(x, training=True)

        gamma = tf.reshape(self.gamma, (-1, 1, 1, self.num_features))
        alpha = tf.reshape(self.alpha, (-1, 1, 1, self.num_features))
        beta = tf.reshape(self.beta, (-1, 1, 1, self.num_features))

        out = gamma * h + means * alpha + beta
        return out


class RefineNetDilated(tfk.layers.Layer):
    def __init__(self, data_shape, ngf, sigmas, logit_transform=False):
        super(RefineNetDilated, self).__init__()
        self.logit_transform = logit_transform
        self.sigmas = sigmas
        self.norm = InstanceNorm2dPlus
        self.ngf = ngf
        self.act = act = tf.nn.elu
        self.data_shape = data_shape

        self.begin_conv = tfk.layers.Conv2D(ngf, 3, strides=1, padding='same',
                                            input_shape=data_shape, name="begin_conv")
        self.normalizer = self.norm(ngf, name='normalizer')

        self.end_conv = tfk.layers.Conv2D(
            data_shape[-1], 3, strides=1, padding='same', name='end_conv')

        self.res1 = [
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm, name="Res1_1"),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm, name="Res1_2")]

        self.res2 = [
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, name="Res2_1"),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, name="Res2_2")]

        self.res3 = [
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2, name="Res3_1"),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2, name="Res3_2")]

        self.res4 = [
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=4, name="Res4_1"),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4, name="Res4_2")]

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True, name="refine1")
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act, name="refine2")
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act, name="refine3")
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True, name="refine4")

    def _compute_cond_module(self, module, x, training=True):
        for m in module:
            x = m(x, training=training)
        return x

    def call(self, inputs, training=True):
        x, y = inputs[0], inputs[1]

        # if not self.logit_transform:
        #     x = 2. * x - 1.

        res_input = self.begin_conv(x)

        layer1 = self._compute_cond_module(self.res1, res_input, training=training)
        layer2 = self._compute_cond_module(self.res2, layer1, training=training)
        layer3 = self._compute_cond_module(self.res3, layer2, training=training)
        layer4 = self._compute_cond_module(self.res4, layer3, training=training)

        ref1 = self.refine1([layer4], layer4.shape[1:3])
        ref2 = self.refine2([layer3, ref1], layer3.shape[1:3])
        ref3 = self.refine3([layer2, ref2], layer2.shape[1:3])
        output = self.refine4([layer1, ref3], layer1.shape[1:3])

        output = self.normalizer(output, training=training)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = tf.gather(params=self.sigmas, indices=y)
        output = output / tf.reshape(used_sigmas, shape=(-1, 1, 1, 1))

        return output

    def get_config(self):
        return {"data_shape": self.data_shape,
                "ngf": self.ngf,
                "logit_transform": self.logit_transform}


class RefineNetDilatedDeeper(tfk.layers.Layer):
    def __init__(self, data_shape, ngf, sigmas, logit_transform=False):
        super(RefineNetDilatedDeeper, self).__init__()
        self.logit_transform = logit_transform
        self.sigmas = sigmas
        self.norm = InstanceNorm2dPlus
        self.ngf = ngf
        self.act = act = tf.nn.elu
        self.data_shape = data_shape

        self.begin_conv = tfk.layers.Conv2D(ngf, 3, strides=1, padding='same',
                                            input_shape=data_shape, name="begin_conv")
        self.normalizer = self.norm(ngf, name='normalizer')

        self.end_conv = tfk.layers.Conv2D(
            data_shape[-1], 3, strides=1, padding='same', name='end_conv')

        self.res1 = [
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm, name="Res1_1"),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm, name="Res1_2")]

        self.res2 = [
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, name="Res2_1"),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, name="Res2_2")]

        self.res3 = [
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, name="Res3_1"),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, name="Res3_2")]

        self.res4 = [
            ResidualBlock(2 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2, name="Res4_1"),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2, name="Res4_2")]

        self.res5 = [
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=4, name="Res4_1"),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4, name="Res4_2")]

        self.refine1 = RefineBlock([4 * self.ngf], 4 * self.ngf, act=act, start=True, name="refine1")
        self.refine2 = RefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, act=act, name="refine2")
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act, name="refine3")
        self.refine4 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act, name="refine4")
        self.refine5 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True, name="refine5")

    def _compute_cond_module(self, module, x, training=True):
        for m in module:
            x = m(x, training=training)
        return x

    def call(self, inputs, training=True):
        x, y = inputs[0], inputs[1]

        # if not self.logit_transform:
        #     x = 2. * x - 1.

        output = self.begin_conv(x)

        layer1 = self._compute_cond_module(self.res1, output, training=training)
        layer2 = self._compute_cond_module(self.res2, layer1, training=training)
        layer3 = self._compute_cond_module(self.res3, layer2, training=training)
        layer4 = self._compute_cond_module(self.res4, layer3, training=training)
        layer5 = self._compute_cond_module(self.res5, layer4, training=training)

        ref1 = self.refine1([layer5], layer5.shape[1:3])
        ref2 = self.refine2([layer4, ref1], layer4.shape[1:3])
        ref3 = self.refine3([layer3, ref2], layer3.shape[1:3])
        ref4 = self.refine4([layer2, ref3], layer2.shape[1:3])
        output = self.refine5([layer1, ref4], layer1.shape[1:3])

        output = self.normalizer(output, training=training)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = tf.gather(self.sigmas, y)
        output = output / tf.reshape(used_sigmas, shape=(-1, 1, 1, 1))

        return output

    def get_config(self):
        return {"data_shape": self.data_shape,
                "ngf": self.ngf,
                "logit_transform": self.logit_transform}


def get_uncompiled_model_v2(args, sigmas):
    # inputs
    perturbed_X = tfk.Input(shape=args.data_shape, dtype=tf.float32, name="perturbed_X")
    sigma_idx = tfk.Input(shape=[], dtype=tf.int32, name="sigma_idx")
    # outputs
    outputs = RefineNetDilated(args.data_shape, args.n_filters,
                               sigmas, args.use_logit)([perturbed_X, sigma_idx])
    # model
    model = tfk.Model(inputs=[perturbed_X, sigma_idx], outputs=outputs, name="ScoreNetwork")

    return model

