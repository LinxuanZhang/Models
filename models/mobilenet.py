'''
MobileNet
https://arxiv.org/pdf/1704.04861.pdf
'''
import tensorflow as tf


class depthwise_block(tf.keras.Model):
    def __init__(self, pointwise_filter, alpha, resolution_multiplier, strides=(1, 1)):
        super().__init__()
        self.pointwise_filter = int(pointwise_filter*alpha)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D((3, 3),
                                                              strides=strides,
                                                              padding='same' if strides == (1, 1) else 'valid',
                                                              depth_multiplier=resolution_multiplier,
                                                              use_bias=False)
        self.pointwise_conv = tf.keras.layers.Conv2D(pointwise_filter, (1, 1), strides = (1, 1), padding='same', use_bias=False)
        self.activation = tf.keras.layers.Activation('relu')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.strides = strides
        self.padding = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)))
    def call(self, input):
        if self.strides == (1, 1):
            x = input
        else:
            x = self.padding(x)
        x = self.depthwise_conv(x)
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.pointwise_conv(x)
        x = self.bn_2(x)
        x = self.activation(x)
        return x


class MobileNetV1(tf.keras.Model):
    def __init__(self, alpha=1, resolution_multiplier=1, num_classes=1000):
        super().__init__()
        self.activation = tf.keras.layers.Activation('relu')
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, tf.keras.layers.Activation='softmax')
        self.conv_1 = tf.keras.layers.Conv2D(filters=int(32*alpha), kernel_size=(3, 3), strides=(1, 1), padding='same',use_bias=False)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.dcb_1 = depthwise_block(64, alpha, resolution_multiplier)
        self.dcb_2 = depthwise_block(128, alpha, resolution_multiplier, strides=(2,2))
        self.dcb_3 = depthwise_block(128, alpha, resolution_multiplier)
        self.dcb_4 = depthwise_block(256, alpha, resolution_multiplier, strides=(2,2))
        self.dcb_5 = depthwise_block(256, alpha, resolution_multiplier)
        self.dcb_6 = depthwise_block(512, alpha, resolution_multiplier, strides=(2,2))
        for i in range(5):
            vars(self)[f'dcb512_{i}'] = depthwise_block(512, alpha, resolution_multiplier)
        self.dcb_7 = depthwise_block(1024, alpha, resolution_multiplier, strides=(2,2))
        self.dcb_8 = depthwise_block(1024, alpha, resolution_multiplier)

    def call(self, input):
        x = self.conv_1(input)
        x = self.bn_1(x)
        x = self.activation(x)

        x = self.dcb_1(x)
        x = self.dcb_2(x)
        x = self.dcb_3(x)
        x = self.dcb_4(x)
        x = self.dcb_5(x)
        x = self.dcb_6(x)

        for i in range(5):
            x = vars(self)[f'dcb512_{i}'](x)

        x = self.dcb_7(x)
        x = self.dcb_8(x)

        x = self.global_pool(x)
        x = self.classifier(x)

        return x
