import numpy as np
import tensorflow as tf
import random as rn
import os
import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
import platform
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.losses import binary_crossentropy, categorical_crossentropy

from psnet_model import superps


np.random.seed(2021)
tf.set_random_seed(2021)
rn.seed(2021)


if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')

if os.name == 'nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')

        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc')
        plt.legend(loc="lower right")

        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')

        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()



# learning rate setting
def scheduler(epoch):
    if epoch > 9 and epoch <= 24:

        K.set_value(superPSmodel.optimizer.lr, 0.0001)
        lr = K.get_value(superPSmodel.optimizer.lr)
        print("lr changed to {}".format(lr))
    elif epoch > 24:
        K.set_value(superPSmodel.optimizer.lr, 0.00001)
        lr = K.get_value(superPSmodel.optimizer.lr)
        print("lr changed to {}".format(lr))
    else:
        lr = K.get_value(superPSmodel.optimizer.lr)
        print("lr changed to {}".format(lr))

    return K.get_value(superPSmodel.optimizer.lr)



def to_categorical_mask(multi_label, nClasses):
    categorical_mask = np.zeros((multi_label.shape[0], multi_label.shape[1], nClasses))
    for c in range(nClasses):
        categorical_mask[:, :, c] = (multi_label == c).astype(int)
    categorical_mask = np.reshape(categorical_mask, (multi_label.shape[0] * multi_label.shape[1], nClasses))
    return categorical_mask

# load data
datapath = r'./gp_data'
img_a = np.load(os.path.join(datapath, 'data_array_a5.npy')) #superpixel number: 500
img_b = np.load(os.path.join(datapath, 'data_array_b2.npy')) #superpixel number: 2000
img_c = np.load(os.path.join(datapath, 'data_array_c.npy'))  #Original, no superpixel pooling
multi_label = np.load(os.path.join(datapath, 'label_array_c.npy')) #label

img_a = img_a.astype(np.float32)
img_a = img_a/255.0
img_b = img_b.astype(np.float32)
img_b = img_b/255.0
img_c = img_c.astype(np.float32)
img_c = img_c/255.0

nClasses = 7
masks = np.zeros((multi_label.shape[0], multi_label.shape[1] * multi_label.shape[2], nClasses))
for i in range(multi_label.shape[0]):
    masks[i, :, :] = to_categorical_mask(multi_label[i], nClasses)


imga = Input(shape=(384, 384, 3))
imgb = Input(shape=(384, 384, 3))
imgc = Input(shape=(384, 384, 3))

o = superps(7, imga, imgb, imgc)
o_shape = Model([imga, imgb, imgc], o).output_shape
output_height = o_shape[1]
output_width = o_shape[2]

o = (Reshape((output_height*output_width, -1)))(o)
o = (Activation('softmax'))(o)
superPSmodel = Model([imga, imgb, imgc], o)
superPSmodel.summary()



history = LossHistory()
adam = Adam(lr=0.001)
superPSmodel.compile(loss=categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

modelpath = r'./model_checkpoint'
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

model_checkpoint = ModelCheckpoint(os.path.join(modelpath, 'psNet_adam_val_C_Sup52_res50_conv128_e' + '{epoch:02d}' + '.h5'), monitor='val_acc', save_best_only=True)
callbacks = [history, LearningRateScheduler(scheduler), model_checkpoint]
hist = superPSmodel.fit([img_a, img_b, img_c], masks, epochs=40, batch_size=8, validation_split=0.1, verbose=1, shuffle=True, callbacks=callbacks)
superPSmodel.save(os.path.join(modelpath, 'psNet_adam_val_C_Sup52_res50_conv128_e40.h5'))

history.loss_plot('epoch')

