# Pyramid Pooling Module is from 'Pyramid Scene Parsing Network  CVPR 2017'
# https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/pspnet.py


import numpy as np
import keras
from keras.models import *
from keras.layers import *
import keras.backend as K

from resnet_multi import superRes50_mul

def resize_image(inp,  s):
    return Lambda(lambda x: K.resize_images(x, height_factor=s[0], width_factor=s[1], data_format='channels_last', interpolation='bilinear'))(inp)

def pool_block(feats, pool_factor):
    h = K.int_shape(feats)[1]
    w = K.int_shape(feats)[2]

    pool_size = strides = [
        int(np.round(float(h) / pool_factor)),
        int(np.round(float(w) / pool_factor))]

    x = AveragePooling2D(pool_size, strides=strides, padding='same')(feats)
    x = resize_image(x, strides)

    return x

def superps(n_classes, img_a, img_b, img_c):
    o = superRes50_mul(img_a, img_b, img_c)

    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]

    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    o = Concatenate(axis=-1)(pool_outs)

    o = Conv2D(128, (1, 1), use_bias=False, kernel_initializer='he_uniform')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(n_classes, (3, 3), padding='same', use_bias=False, kernel_initializer='he_uniform')(o)
    o = resize_image(o, (8, 8))

    return o





