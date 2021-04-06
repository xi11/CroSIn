
import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import math
from glob import glob
import time

from keras import backend as K
import matplotlib
import platform
from keras.models import load_model
from keras.losses import binary_crossentropy, categorical_crossentropy

if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')

if os.name=='nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

K.set_image_data_format('channels_last')



#openCV: BGR
color_coder = [(0, 0, 0), (0, 255, 0), (255, 0, 255), (0, 0, 128), (0, 255, 255), (0, 0, 255), (255, 0, 0)]

def get_colored_segmentation_image(seg_arr, n_classes, colors=color_coder):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img



class Patches:
    def __init__(self, img_patch_h, img_patch_w, stride_h=384, stride_w=384, label_patch_h=None, label_patch_w=None):
        assert img_patch_h > 0, 'Height of Image Patch should be greater than 0'
        assert img_patch_w > 0, 'Width of Image Patch should be greater than 0'
        assert label_patch_h > 0, 'Height of Label Patch should be greater than 0'
        assert label_patch_w > 0, 'Width of Label Patch should be greater than 0'
        assert img_patch_h >= label_patch_h, 'Height of Image Patch should be greater or equal to Label Patch'
        assert img_patch_w >= label_patch_w, 'Width of Image Patch should be greater or equal to Label Patch'
        assert stride_h > 0, 'Stride should be greater than 0'
        assert stride_w > 0, 'Stride should be greater than 0'
        assert stride_h <= label_patch_h, 'Row Stride should be less than or equal to Label Patch Height'
        assert stride_w <= label_patch_w, 'Column Stride should be less than or equal to Label Patch Width'
        self.img_patch_h = img_patch_h
        self.img_patch_w = img_patch_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.label_patch_h = label_patch_h
        self.label_patch_w = label_patch_w
        self.img_h = None
        self.img_w = None
        self.img_d = None
        self.num_patches_img = None
        self.num_patches_img_h = None
        self.num_patches_img_w = None
        self.label_diff_pad_h = 0
        self.label_diff_pad_w = 0
        self.pad_h = 0
        self.pad_w = 0

    @staticmethod
    def read_image(input_str):
        image = np.array(Image.open(input_str))
        return image

    def update_variables(self, image):
        self.img_h = np.size(image, 0)
        self.img_w = np.size(image, 1)
        self.img_d = np.size(image, 2)

    def extract_patches_img_label(self, input_img_value):
        if type(input_img_value) == str:
            image = self.read_image(input_img_value)
        elif type(input_img_value) == np.ndarray:
            image = input_img_value
        else:
            raise Exception('Please input correct image path or numpy array')
        self.update_variables(image)

        img_patch_h = self.img_patch_h
        img_patch_w = self.img_patch_w

        stride_h = self.stride_h
        stride_w = self.stride_w

        if image.shape[0] < img_patch_h:
            self.pad_h = img_patch_h - image.shape[0]
        else:
            self.pad_h = 0

        if image.shape[1] < img_patch_w:
            self.pad_w = img_patch_w - image.shape[1]
        else:
            self.pad_w = 0

        image = np.lib.pad(image, ((self.pad_h, self.pad_h), (self.pad_w, self.pad_w), (0, 0)),'symmetric')
        self.update_variables(image)

        img_h = self.img_h
        img_w = self.img_w
        print(img_h, img_w)

        self.num_patches_img_h = math.ceil((img_h - img_patch_h) / stride_h + 1)
        self.num_patches_img_w = math.ceil(((img_w - img_patch_w) / stride_w + 1))
        num_patches_img = self.num_patches_img_h*self.num_patches_img_w
        self.num_patches_img = num_patches_img
        iter_tot = 0
        img_patches = np.zeros((num_patches_img, img_patch_h, img_patch_w, image.shape[2]), dtype=image.dtype)
        for h in range(int(math.ceil((img_h - img_patch_h) / stride_h + 1))):
            for w in range(int(math.ceil((img_w - img_patch_w) / stride_w + 1))):
                start_h = h * stride_h
                end_h = (h * stride_h) + img_patch_h
                start_w = w * stride_w
                end_w = (w * stride_w) + img_patch_w
                if end_h > img_h:
                    start_h = img_h - img_patch_h
                    end_h = img_h

                if end_w > img_w:
                    start_w = img_w - img_patch_w
                    end_w = img_w

                img_patches[iter_tot, :, :, :] = image[start_h:end_h, start_w:end_w, :]
                iter_tot += 1

        return img_patches


    def merge_patches(self, patches):
        img_h = self.img_h
        img_w = self.img_w
        img_patch_h = self.img_patch_h
        img_patch_w = self.img_patch_w
        label_patch_h = self.label_patch_h
        label_patch_w = self.label_patch_w
        stride_h = self.stride_h
        stride_w = self.stride_w
        num_patches_img = self.num_patches_img
        assert num_patches_img == patches.shape[0], 'Number of Patches do not match'
        assert img_patch_h == patches.shape[1] or label_patch_h == patches.shape[1], 'Height of Patch does not match'
        assert img_patch_w == patches.shape[2] or label_patch_w == patches.shape[2], 'Width of Patch does not match'

        image = np.zeros((img_h, img_w, patches.shape[3]), dtype=np.float)
        sum_c = np.zeros((img_h, img_w, patches.shape[3]), dtype=np.float)
        iter_tot = 0
        for h in range(int(math.ceil((img_h - img_patch_h) / stride_h + 1))):
            for w in range(int(math.ceil((img_w - img_patch_w) / stride_w + 1))):
                start_h = h * stride_h
                end_h = (h * stride_h) + img_patch_h
                start_w = w * stride_w
                end_w = (w * stride_w) + img_patch_w
                if end_h > img_h:
                    start_h = img_h - img_patch_h
                    end_h = img_h

                if end_w > img_w:
                    start_w = img_w - img_patch_w
                    end_w = img_w

                if self.label_diff_pad_h == 0 and self.label_diff_pad_w == 0:
                    image[start_h:end_h, start_w:end_w, :] += patches[iter_tot, :, :,:]
                    sum_c[start_h:end_h, start_w:end_w, :] += 1.0
                else:
                    image[
                        start_h+self.label_diff_pad_h:start_h + label_patch_h + self.label_diff_pad_h,
                        start_w+self.label_diff_pad_w:start_w + label_patch_w + self.label_diff_pad_w] += \
                        patches[iter_tot, :, :]
                    sum_c[
                        start_h+self.label_diff_pad_h:start_h + label_patch_h + self.label_diff_pad_h,
                        start_w+self.label_diff_pad_w:start_w + label_patch_w + self.label_diff_pad_w] += 1.0
                iter_tot += 1

        if self.pad_h != 0 and self.pad_w != 0:
            sum_c = sum_c[self.pad_h:-self.pad_h, self.pad_w:-self.pad_w, :]
            image = image[self.pad_h:-self.pad_h, self.pad_w:-self.pad_w, :]

        if self.pad_h == 0 and self.pad_w != 0:
            sum_c = sum_c[:, self.pad_w:-self.pad_w, :]
            image = image[:, self.pad_w:-self.pad_w, :]

        if self.pad_h != 0 and self.pad_w == 0:
            sum_c = sum_c[self.pad_h:-self.pad_h, :, :]
            image = image[self.pad_h:-self.pad_h, :, :]

        assert (np.min(sum_c) >= 1.0)
        image = np.divide(image, sum_c)

        return image


modelpath = r'./model_checkpoint'
model=load_model(os.path.join(modelpath, 'psNet_adam_val_C_Sup52_res50_conv128_e40.h5'))
test_img_dir_a = r'./testdata/test_super500'
test_img_dir_b = r'./testdata/test_super2000'
test_img_dir_c = r'./testdata/images'
save_dir = r'./results'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

imgs = sorted(glob(os.path.join(test_img_dir_c, '*.jpg')))
for im_f in imgs:
    img_name = os.path.splitext(os.path.basename(im_f))[0]
    print(img_name)

    testImga = np.array(Image.open(os.path.join(test_img_dir_a, img_name + '.jpg')))
    testImgb = np.array(Image.open(os.path.join(test_img_dir_b, img_name + '.jpg')))
    testImgc = np.array(Image.open(os.path.join(test_img_dir_c, img_name + '.jpg')))
    patch_obj = Patches(img_patch_h=384, img_patch_w=384, stride_h=192, stride_w=192, label_patch_h=384, label_patch_w=384)
    testData_a = patch_obj.extract_patches_img_label(testImga)
    testData_a = testData_a.astype(np.float32)
    testData_a = testData_a / 255.0

    testData_b = patch_obj.extract_patches_img_label(testImgb)
    testData_b = testData_b.astype(np.float32)
    testData_b = testData_b / 255.0

    testData_c = patch_obj.extract_patches_img_label(testImgc)
    testData_c = testData_c.astype(np.float32)
    testData_c = testData_c / 255.0

    outData = model.predict([testData_a, testData_b, testData_c])
    outData = outData.reshape((-1, 384, 384, 7))
    merge_output = patch_obj.merge_patches(outData)
    merge_output = merge_output.argmax(axis=2)

    seg_mask = get_colored_segmentation_image(merge_output, 7, colors=color_coder)
    cv2.imwrite(os.path.join(save_dir, 'mask_' + img_name + '.jpg'), seg_mask)




