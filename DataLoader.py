import glob
import operator
import datetime
import math
# import transformations
import scipy.ndimage as nd
import nibabel as nib
import pandas as pd
# from keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence
from scipy.ndimage import map_coordinates
# from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
from random import gauss
import random


# from transformations import rotation_matrix
# from pytransform3d.rotations import matrix_from_angle
def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example:
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop


def getInformationFromDemographics(id):
    df = pd.read_csv("./demo_new.csv")
    res = df[df['eid'] == int(id)]
    return res['age'].values[0], int(res['sex'].values[0])


def getImageData():
    path = "/home/shulab/liuzhenzhao/biobank/T1/"
    t1Paths = glob.glob(path + "*/T1_linear_to_MNI.nii.gz")
    dataframe = pd.DataFrame(columns=['id', 'path', 'Age', 'Gender'])
    for path in t1Paths:
        id = path.split("/")[6].split("_")[0]
        age, sex = getInformationFromDemographics(id)
        dict = [{'id': id, 'path': path, 'Age': age, 'Gender': sex}]
        dataframe = dataframe.append(dict, ignore_index=True)
    dataframe.to_csv("./data.csv")
    return dataframe


class dataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, features, labels, batch_size=32, meanImg=None, dim=(121, 145, 121), maxAngle=40, maxShift=10,
                 shuffle=True, augment=False, includeGender=False,resize_img=False,normalise_mode=0):
        'Initialization'
        self.batch_size = batch_size
        self.features = features
        self.labels = labels
        self.dim = dim
        self.meanImg = meanImg
        self.augment = augment
        self.maxAngle = maxAngle
        self.maxShift = maxShift
        self.shuffle = shuffle
        self.IncludeGender = includeGender
        self.on_epoch_end()
        self.resize_img=resize_img
        self.normalise_mode=normalise_mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.features[0].shape[0] / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.features[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # print(index)
        index = index % self.__len__()
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # features_temp = [self.features[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, indexes):
        # print("begin generate a batch of data...")
        # 'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, self.dim[0],self.dim[1],self.dim[2]),dtype=np.uint8)
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2], 1))
        age = np.empty((self.batch_size))
        if self.IncludeGender:
            sex = np.empty((self.batch_size))
            # scanner = np.empty((self.batch_size))
        # Generate data
        for i, index in enumerate(indexes):
            X[i, :, :, :, :] = processing(self.features[0][index], self.dim, self.meanImg, resizeImg=self.resize_img,
                                          augment=self.augment,normalise_mode=self.normalise_mode)
            age[i] = self.labels[index]
            #tmp=age[i,:,:,:,0]

            if self.IncludeGender:
                # scanner[i] = self.features[1][index]
                sex[i] = self.features[1][index]
        # print("finished generate a batch of data!")
        if self.IncludeGender:
            return [X, sex], [age]
        else:
            return [X], [age]


class patchedDataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, features, labels, batch_size=32, meanImg=None, dim=(121, 145, 121), maxAngle=40, maxShift=10,
                 shuffle=True, augment=False, includeGender=False):
        'Initialization'
        self.batch_size = batch_size
        self.features = features
        self.labels = labels
        self.dim = dim
        self.meanImg = meanImg
        self.augment = augment
        self.maxAngle = maxAngle
        self.maxShift = maxShift
        self.shuffle = shuffle
        self.IncludeGender = includeGender
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.features[0].shape[0] / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.features[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        index = index % self.__len__()
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, indexes):
        # print("begin generate a batch of data...")
        # 'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, self.dim[0],self.dim[1],self.dim[2]),dtype=np.uint8)
        tmp=np.empty((self.dim[0], self.dim[1], self.dim[2],1))
        patch_tmp=split_to_patches(tmp,12,12,12)
        n_patch=len(patch_tmp)

        X = np.empty((self.batch_size*n_patch, 12, 12, 12, 1))
        age = np.empty((self.batch_size*n_patch))
        if self.IncludeGender:
            sex = np.empty((self.batch_size*n_patch))
            # scanner = np.empty((self.batch_size*n_patch))
        # Generate data
        for i, index in enumerate(indexes):
            #X[i, :, :, :, :] = processing(self.features[0][index], self.dim, self.meanImg, resizeImg=False,augment=self.augment,normalise_mode=2)
            image=processing(self.features[0][index], self.dim, self.meanImg, resizeImg=False, augment=self.augment,
                       normalise_mode=2)
            Patches=split_to_patches(image,12,12,12)
            for k in range(n_patch):
                X[i*n_patch+k]=Patches[k]
            age[i*n_patch:(i+1)*n_patch] = self.labels[index]
            if self.IncludeGender:
                # scanner[i*n_patch:(i+1)*n_patch] = self.features[1][index]
                sex[i*n_patch:(i+1)*n_patch] = self.features[1][index]

        if self.IncludeGender:
            return [X, sex], [age]
        else:
            return [X], [age]



def split_to_patches(vol,delta_x,delta_y,delta_z):
        x_start=-delta_x
        (x_end,y_end,z_end,hold)=vol.shape
        patches=[]
        while x_start<x_end:
            x_start = x_start + delta_x
            y_start=-delta_y
            while y_start<y_end:
                y_start = y_start + delta_y
                z_start=-delta_z
                while z_start<z_end:
                    z_start = z_start + delta_z
                    patch=vol[x_start:x_start+delta_x,y_start:y_start+delta_y,z_start:z_start+delta_z,:]
                    dim=patch.shape
                    patch=np.pad(patch,((0,12-dim[0]),(0,12-dim[1]),(0,12-dim[2]),(0,0)),'constant')
                    patches.append(patch)
        return patches





def resize3d(image, new_shape, order=3):
    real_resize_factor = tuple(map(operator.truediv, new_shape, image.shape))
    return nd.zoom(image, real_resize_factor, order=order)


def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]



def normalise(image,normalise_mode):
    if normalise_mode==1:
        min=image.min()
        max=image.max()
        return 100*(image-min)/(max-min)
    elif normalise_mode==2:
        mean=image.mean()
        var=image.var()
        epsilon=1e-6
        return (image-mean)/(math.sqrt(var+epsilon))
    else:
        return image

def loadMR(path):
    img = nib.load(path).get_fdata()
    #img_affine=nib.load(path).affine
    #img = img.astype(np.float32)
    img = img.astype(np.int)
    return img


def processing(features, inputShape, meanImg, maxAngle=40, maxShift=10, resizeImg=False, augment=False, training=False,normalise_mode=2):
    X_T1 = loadMR(features)
    X_T1 = normalise(X_T1, normalise_mode)
    if meanImg is not None:
        X_T1 = X_T1 - meanImg
    # if augment:
    #     X_T1 = coordinateTransformWrapper(X_T1, maxDeg=maxAngle, maxShift=maxShift)
    if resizeImg:
        # inputShape = (182,218,182)
        # X_T1 = resize3d(X_T1, inputShape)
        #X_T1 = crop_center(X_T1, inputShape)
        X_T1=X_T1[9:82,9:101,0:79]
    #nib.Nifti1Image(X_T1,img_affine).to_filename('./temp.nii.gz')
    return X_T1.reshape(inputShape + (1,))
