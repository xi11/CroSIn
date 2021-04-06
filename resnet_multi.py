import keras
from keras.models import *
from keras.layers import *
from keras import layers

# identity_block and conv_block are from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py



def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with
    strides=(2,2) and the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters



    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)


    x = Conv2D(filters2, kernel_size, padding='same', use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def supres50_a(img_a):
    # x = ZeroPadding2D((6, 6))(img_input)
    x = Conv2D(16, (7, 7), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False,
               kernel_initializer='he_uniform',
               name='conv1a')(img_a)

    x = BatchNormalization(name='bn_conv1a')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [16, 16, 64], stage=2, block='aa', strides=(1, 1))
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='ab')
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='ac')

    Sa = x
    return Sa

def supres50_b(img_b, Sa):
    x = Conv2D(16, (7, 7), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False,
               kernel_initializer='he_uniform',
               name='conv1b')(img_b)

    x = BatchNormalization(name='bn_conv1b')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [16, 16, 64], stage=2, block='ba', strides=(1, 1))
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='bb')
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='bc')

    f = Concatenate(axis=-1)([x, Sa])

    x = conv_block(f, 3, [32, 32, 128], stage=3, block='ba')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='bb')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='ac')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='ad')

    Sb = x

    return Sb

def supres50_c(img_c, Sb):
    x = Conv2D(16, (7, 7), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False,
               kernel_initializer='he_uniform', name='conv1c')(img_c)

    x = BatchNormalization(name='bn_conv1c')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [16, 16, 64], stage=2, block='ca', strides=(1, 1))
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='cb')
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='cc')
    #f1 = Concatenate(axis=-1)([x, Sa])

    x = conv_block(x, 3, [32, 32, 128], stage=3, block='ca')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='cb')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='cc')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='cd')
    f2 = Concatenate(axis=-1)([x, Sb])

    x = conv_block(f2, 3, [64, 64, 256], stage=4, block='ca')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='cb')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='cc')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='cd')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='ce')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='cf')
    #f4 = one_side_pad(x)
    Sc = x

    return Sc

def superRes50_mul(img_a, img_b, img_c):
    Sa = supres50_a(img_a)
    Sb = supres50_b(img_b, Sa)
    Sc = supres50_c(img_c, Sb)
    return Sc