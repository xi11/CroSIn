
from scipy.io import savemat, loadmat
import os
import pathlib
import numpy as np
from PIL import Image
import math


def read_img(data_file, label_file):    # to be checked with if file or path
    data = np.array(Image.open(data_file))
    labels = loadmat(label_file)['mask']   # label is in .mat format
    return data, labels

def get_label_file(data_file,label_path): #get mask with corresponding name
    data_file_name = data_file.split('\\')[-1]  #split path and get the last one, which is the file name
    print(data_file_name)
    label_file_name = data_file_name.split('.')[0] + '.mat' #split the data name and get the name before ., combined with '.mat'
    label_file = os.path.join(label_path, label_file_name)

    return label_file

def write_to_npy(data_files, label_path, save_path):
    patch_obj = Patches(img_patch_h=384, img_patch_w=384, stride_h=192, stride_w=192, label_patch_h=384, label_patch_w=384)
    data_array = []
    label_array = []

    for file_n in range(0, len(data_files)):

        curr_data_file = str(data_files[file_n])
        curr_label_file = get_label_file(curr_data_file,label_path)

        data, labels = read_img(curr_data_file, curr_label_file)
        data, labels = patch_obj.extract_patches_img_label(data, labels)


        print('shape:', data.shape, labels.shape)
        print(type(data), type(labels))

        data_array.append(data)
        label_array.append(labels)

    data_array = np.asarray(data_array)
    label_array = np.asarray(label_array)
    data_array = np.vstack(data_array)
    label_array = np.vstack(label_array)
    print('shape: %s, data type: %s' %(data_array.shape, data_array.dtype))
    np.save(os.path.join(save_path, 'data_array_c.npy'), data_array) #train patches, in .npy format with a size of 384*384
    np.save(os.path.join(save_path, 'label_array_c.npy'), label_array) #train labels, in .npy format with a size of 384*384





#Patches
class Patches:
    def __init__(self, img_patch_h, img_patch_w, stride_h=1, stride_w=1, label_patch_h=None, label_patch_w=None):
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

    def extract_patches_img_label(self, input_img_value, input_label_value):
        if type(input_img_value) == str:
            image = self.read_image(input_img_value)
        elif type(input_img_value) == np.ndarray:
            image = input_img_value
        else:
            raise Exception('Please input correct image path or numpy array')
        self.update_variables(image)

        if type(input_label_value) == str:
            label = loadmat(input_label_value)['mask']
        elif type(input_label_value) == np.ndarray:
            label = input_label_value
        else:
            raise Exception('Please input correct label path or numpy array')
        assert image.shape[:2] == label.shape, 'Image and Label shape should be the same'

        img_patch_h = self.img_patch_h
        img_patch_w = self.img_patch_w
        label_patch_h = self.label_patch_h
        label_patch_w = self.label_patch_w
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
        label = np.lib.pad(label, ((self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), 'symmetric')


        self.label_diff_pad_h = math.ceil((img_patch_h - label_patch_h) / 2.0)
        self.label_diff_pad_w = math.ceil((img_patch_w - label_patch_w) / 2.0)

        image = np.lib.pad(image, ((self.label_diff_pad_h, self.label_diff_pad_h), (self.label_diff_pad_w, self.label_diff_pad_w), (0, 0)), 'symmetric')
        label = np.lib.pad(label, ((self.label_diff_pad_h, self.label_diff_pad_h), (self.label_diff_pad_w, self.label_diff_pad_w)), 'symmetric')

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
        label_patches = np.zeros((num_patches_img, label_patch_h, label_patch_w), dtype=image.dtype)
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
                label_patches[iter_tot, :, :] = label[start_h:end_h, start_w:end_w]
                iter_tot += 1

        return img_patches, label_patches



def run(opts_in):
    save_path = opts_in['save_path']
    data_path = opts_in['data_path']
    label_path = opts_in['label_path']
    print(label_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_data = list(data_path.glob('*.jpg'))

    print('data number: ', len(train_data))

    write_to_npy(data_files = train_data, label_path = label_path, save_path=save_path)



if __name__ == '__main__':
    opts = {
        'save_path': pathlib.Path(r'./gp_data'),           #train data, in .npy format with a size of 384*384
        'data_path': pathlib.Path(r'./traindata/images'),  #train images
        'label_path': pathlib.Path(r'./traindata/mat'),    #masks in .mat format

    }

    run(opts_in=opts)