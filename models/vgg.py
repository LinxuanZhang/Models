'''
VGG
https://arxiv.org/pdf/1409.1556v6.pdf
'''
import tensorflow as tf

block_args = {
    'A':[(64, 3, 1), (128, 3, 1), (256, 3, 2), (512, 3, 2), (512, 3, 2)],
    'B':[(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2), (512, 3, 2)],
    'C':[(64, 3, 2), (128, 3, 2), (256, [3, 3, 1], 3), (512, [3, 3, 1], 3), (512, [3, 3, 1], 3)],
    'D':[(64, 3, 2), (128, 3, 2), (256, 3, 3), (512, 3, 3), (512, 3, 3)],
    'E':[(64, 3, 2), (128, 3, 2), (256, 3, 4), (512, 3, 4), (512, 3, 4)]
}


class vgg_blocks(tf.keras.Model):
    def __init__(self, filters, kernel_sizes, repetitions, pool_size = 2, strides = 2):
        '''
        kernel_sizes need to be either integer or list of integer
        '''
        super().__init__()
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.repetitions = repetitions

        if isinstance(kernel_sizes, list):
            for i in range(repetitions):
                vars(self)[f'block_{i}'] = tf.keras.layers.Conv2D(filters, kernel_sizes[i], activation='relu', padding='same')
        elif isinstance(kernel_sizes, int):
            for i in range(repetitions):
                vars(self)[f'block_{i}'] = tf.keras.layers.Conv2D(filters, kernel_sizes, activation='relu', padding='same')

        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), strides=(strides, strides))

    def call(self, input):
        x = self.block_0(input)

        for i in range(1, self.repetitions):
            x = vars(self)[f'block_{i}'](x)

        x = self.max_pool(x)
        return x


class vgg_feature(tf.keras.Model):
    def __init__(self, block_arg):
        super().__init__()
            self.block_0 = vgg_blocks(block_arg[0])
            self.block_1 = vgg_blocks(block_arg[1])
            self.block_2 = vgg_blocks(block_arg[2])
            self.block_3 = vgg_blocks(block_arg[3])
            self.block_4 = vgg_blocks(block_arg[4])

    def call(self, input):
        x = self.block_0(input)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        return x


class VGG(tf.keras.Model):
    def __init__(self, block_arg, num_classes=1000, tf.keras.layers.Dropout_ratio=0.5):
        super().__init__()
        self.feature = vgg_feature(block_arg)
        self.Flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu')
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu')
        self.Dropout_1 = tf.keras.layers.Dropout(tf.keras.layers.Dropout_ratio)
        self.Dropout_2 = tf.keras.layers.Dropout(tf.keras.layers.Dropout_ratio)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, input):
        x = self.feature(input)
        x = self.Flatten(x)
        x = self.fc1(x)
        x = self.Dropout_1(x)
        x = self.fc2(x)
        x = self.Dropout_2(x)
        x = self.classifier(x)
    return x


def _vgg(block_arg, **kwargs):
    tf.keras.Model = VGG(block_arg, **kwargs)
    return tf.keras.Model


def vgg11(**kwargs):
    return _vgg(block_args['A'])


def vgg13(**kwargs):
    return _vgg(block_args['B'])


def vgg16_1(**kwargs):
    return _vgg(block_args['C'])


def vgg16(**kwargs):
    return _vgg(block_args['D'])


def vgg19(**kwargs):
    return _vgg(block_args['E'])
