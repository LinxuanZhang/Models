'''
ResNet
https://arxiv.org/pdf/1512.03385v1.pdf
'''
import tensorflow as tf

block_args = {
    '18':[('bb', 64, 2), ('bb', 128, 2), ('bb', 256, 2), ('bb', 512, 2)],
    '34':[('bb', 64, 3), ('bb', 128, 4), ('bb', 256, 6), ('bb', 512, 3)],
    '50':[('bn', [64, 64, 25], 3), ('bn', [128, 128, 512], 4), ('bn', [256, 256, 1024], 6), ('bn', [512, 512, 2048], 3)],
    '101':[('bn', [64, 64, 25], 3), ('bn', [128, 128, 512], 4), ('bn', [256, 256, 1024], 23), ('bn', [512, 512, 2048], 3)],
    '152':[('bn', [64, 64, 25], 3), ('bn', [128, 128, 512], 4), ('bn', [256, 256, 1024], 36), ('bn', [512, 512, 2048], 3)]
}

class basic_blocks(tf.keras.Model):
    def __init__(self, filter):
        '''
        filter:int
        '''
        super().__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filter, 3, ptf.keras.layers.Adding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(filter, 3, ptf.keras.layers.Adding='same')
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()

    def call(self, input):
        x = self.conv_1(input)
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.add([x, input])
        x = self.activation(x)
        return x


class bottleneck(tf.keras.Model):
    def __init__(self, filters):
        '''
        filters: list of 3 int
        '''
        super().__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filters[0], 1, ptf.keras.layers.Adding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(filters[1], 3, ptf.keras.layers.Adding='same')
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.conv_3 = tf.keras.layers.Conv2D(filters[2], 1, ptf.keras.layers.Adding='same')
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()

    def call(self, input):
        x = self.conv_1(input)
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.activation(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.add([x, input])
        x = self.activation(x)
        return x


class identity_blocks(tf.keras.Model):
    def __init__(self, block_arg):
        super().__init__()
        self.repetitions = block_arg[2]
        if block_arg[0] == 'bb':
            for i in range(block_arg[2]):
                vars(self)[f'building_block{i}'] = basic_blocks(block_arg[1])
        else if block_arg[0] == 'bn':
            for i in range(block_arg[2]):
                vars(self)[f'building_block{i}'] = bottleneck(block_arg[1])
        else:
            raise ValueError("block arg needs to be either bb or bn")

    def call(self, input):
        x = self.building_block0(input)
        for i in range(1, self.repetitions):
            x = vars(self)[f'building_block{i}'](x)
        return x


class ResNet(tf.keras.Model):
    def __init__(self, block_arg, num_classes=1000):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(64, 7, ptf.keras.layers.Adding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPooling2D((3, 3))

        self.id_block1 = identity_blocks(block_arg[0])
        self.id_block2 = identity_blocks(block_arg[1])
        self.id_block3 = identity_blocks(block_arg[2])
        self.id_block4 = identity_blocks(block_arg[3])

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, tf.keras.layers.Activation='softmax')

    def call(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = self.id_block1(x)
        x = self.id_block2(x)
        x = self.id_block3(x)
        x = self.id_block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def _resnet(block_arg, **kwargs):
    Model = ResNet(block_arg, **kwargs)
    return Model


def resnet18(**kwargs):
    return _resnet(block_args['18'])


def resnet34(**kwargs):
    return _resnet(block_args['34'])


def resnet50(**kwargs):
    return _resnet(block_args['50'])


def resnet101(**kwargs):
    return _resnet(block_args['101'])


def resnet152(**kwargs):
    return _resnet(block_args['152'])
