'''
ConvNeXt
https://arxiv.org/abs/2201.03545
'''
import tensorflow as tf

block_args = {
    'convnext_tiny':{depths=[3, 3, 9, 3],dims=[96, 192, 384, 768]},
    'convnext_small':{depths=[3, 3, 27, 3],dims=[96, 192, 384, 768]},
    'convnext_base':{depths=[3, 3, 27, 3],dims=[128, 256, 512, 1024]},
    'convnext_large':{depths=[3, 3, 27, 3],dims=[192, 384, 768, 1536]},
    'convnext_xlarge':{depths=[3, 3, 27, 3],dims=[256, 512, 1024, 2048]}
}

class convnext_block(tf.keras.Model):
    def __init__(self, dim, drop_path_rate=0, layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(7, 7), padding='same')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pointwise_linear1 = tf.keras.layers.Dense(4*dim)
        self.pointwise_linear2 = tf.keras.layers.Dense(dim)
        self.activation = tf.keras.layers.Activation('gelu')
        self.gamma = tf.Variable(layer_scale_init_value*tf.ones(dim), trainable=True) if layer_scale_init_value>0 else None
        self.drop_path_rate = drop_path_rate

    def call(self, input, training=False):
        x = self.depthwise_conv(input)
        x = tf.transpose(x, [0, 2, 3, 1])
        x = self.layernorm(x)
        x = self.pointwise_linear1(x)
        x = self.activation(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = tf.transpose(x, [0, 3, 2, 1])
        x = input + self.drop_path(x, self.drop_path_rate, traning)
        return x

    def drop_path(self, input, drop_path_rate, training=False):
        if (not training) or (drop_path_rate == 0.):
            return input
        shape = (tf.shape(input)[0],) + (1,)*(len(tf.shape(input)) - 1)
        keep = tf.floor(1-drop_path_rate+tf.random.uniform(shape, dtype=input.dtype))
        res = tf.math.divide(input, 1-drop_path_rate)*keep
        return res


class ConvNeXt(tf.keras.Model):
    def __init__(self, num_classes=1000,
                 depths=[3,3,9,3],
                 dims=[96, 192, 384, 768], #(depths, dims)
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.
                 include_top = True):
        super().__init__()
        self.stem = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dims[0], kernel_size=4, strides=4, padding='same')
            tf.keras.LayerNormalization(epsilon=1e-6)
            ])
        for i in range(1, 4):
            vars(self)[f'downsample_layers{i}'] = tf.keras.Sequential([
            tf.keras.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Conv2D(dims[i], kernel_size=2, strides=2, padding='same')
            ])
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            vars(self)[f'stage{i}'] = tf.keras.Sequential([
            convnext_block(dim=dims[i], drop_path_rate=dp_rates[cur+j], layer_scale_init_value=layer_scale_init_value)
            for j in range(depths[i])
            ])
            cur += depths[i]
        self.include_top = include_top
        if include_top:
            self.head = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D()
            tf.keras.layers.LayerNormalization(epsilon=1e-6)
            tf.keras.layers.Dense(num_classes, tf.keras.layers.Activation='softmax')
            ])
        else:
            self.head = None

    def call(self, input):
        x = self.stem(x)
        x = self.stage0(x)
        for i in range(1, 4):
            x = vars(self)[f'downsample_layers{i}'](x)
            x = vars(self)[f'stage{i}'](x)
        if self.include_top:
            x = self.head(x)
        retrun x


def _convnext(block_args, **kwargs):
    Model = ConvNeXt(depths=block_args['depths'], dims=block_args['dims'])
    return Model


def convnext_tiny(**kwargs):
    return _convnext(block_args['convnext_tiny'])


def convnext_small(**kwargs):
    return _convnext(block_args['convnext_small'])


def convnext_base(**kwargs):
    return _convnext(block_args['convnext_base'])


def convnext_large(**kwargs):
    return _convnext(block_args['convnext_large'])


def convnext_xlarge(**kwargs):
    return _convnext(block_args['convnext_xlarge'])
