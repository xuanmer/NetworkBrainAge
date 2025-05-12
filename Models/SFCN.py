from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, add, concatenate, \
    GlobalAveragePooling3D, Conv3D, LeakyReLU, ELU, MaxPooling3D, AveragePooling3D
from tensorflow.keras.regularizers import l2 as L2
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam, SGD, Adagrad
# import tensorflow.compat.v1 as tf
import tensorflow as tf

#tf.disable_v2_behavior()
import pandas as pd
import os
import scipy.ndimage as nd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from tensorflow.keras.models import load_model
import random
import glob
import nibabel as nib
from collections import defaultdict


def regression_hinge(y_true, y_pred):
    epsilon = 2
    return K.mean(K.maximum(K.abs(y_true - y_pred) - epsilon, 0.), axis=-1)


def generateAgePredictionSFCNet(inputShape, paddingType='same', initType='he_uniform', regAmount=0.00005, dropRate=0.5,
                                includeGender=True):
    t1Input = Input(inputShape + (1,), name='T1_Img')

    with tf.name_scope('layer0'):
        inputs = t1Input
        features = 8
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType, kernel_regularizer=L2(regAmount),
                        kernel_initializer=initType)(inputs)
        hidden = BatchNorm(renorm=True)(hidden)
        # hidden = ELU(alpha=1.0)(hidden)
        # hidden = Conv3D(features, (3, 3, 3), padding=paddingType, kernel_regularizer=L2(regAmount),
        #                 kernel_initializer=initType)(hidden)
        # hidden = BatchNorm(renorm=True)(hidden)
        # shortcut = Conv3D(features, (1, 1, 1), strides=(1, 1, 1), padding=paddingType, kernel_initializer=initType)(
        #     inputs)
        # hidden = add([shortcut, hidden])
        outputs = ELU(alpha=1.0)(hidden)

    pooling = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding=paddingType)(outputs)

    with tf.name_scope('layer1'):
        inputs = pooling
        features = 16
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType, kernel_regularizer=L2(regAmount),
                        kernel_initializer=initType)(inputs)
        hidden = BatchNorm(renorm=True)(hidden)
        # hidden = ELU(alpha=1.0)(hidden)
        # hidden = Conv3D(features, (3, 3, 3), padding=paddingType, kernel_regularizer=L2(regAmount),
        #                 kernel_initializer=initType)(hidden)
        # hidden = BatchNorm(renorm=True)(hidden)
        # shortcut = Conv3D(features, (1, 1, 1), strides=(1, 1, 1), padding=paddingType, kernel_initializer=initType)(
        #     inputs)
        # hidden = add([shortcut, hidden])
        outputs = ELU(alpha=1.0)(hidden)

    pooling = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding=paddingType)(outputs)

    with tf.name_scope('layer2'):
        inputs = pooling
        features = 32
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType, kernel_regularizer=L2(regAmount),
                        kernel_initializer=initType)(inputs)
        hidden = BatchNorm(renorm=True)(hidden)
        # hidden = ELU(alpha=1.0)(hidden)
        # hidden = Conv3D(features, (3, 3, 3), padding=paddingType, kernel_regularizer=L2(regAmount),
        #                 kernel_initializer=initType)(hidden)
        # hidden = BatchNorm(renorm=True)(hidden)
        # shortcut = Conv3D(features, (1, 1, 1), strides=(1, 1, 1), padding=paddingType, kernel_initializer=initType)(
        #     inputs)
        # hidden = add([shortcut, hidden])
        outputs = ELU(alpha=1.0)(hidden)

    pooling = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding=paddingType)(outputs)

    with tf.name_scope('layer3'):
        inputs = pooling
        features = 32
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType, kernel_regularizer=L2(regAmount),
                        kernel_initializer=initType)(inputs)
        hidden = BatchNorm(renorm=True)(hidden)
        # hidden = ELU(alpha=1.0)(hidden)
        # hidden = Conv3D(features, (3, 3, 3), padding=paddingType, kernel_regularizer=L2(regAmount),
        #                 kernel_initializer=initType)(hidden)
        # hidden = BatchNorm(renorm=True)(hidden)
        # shortcut = Conv3D(features, (1, 1, 1), strides=(1, 1, 1), padding=paddingType, kernel_initializer=initType)(
        #     inputs)
        # hidden = add([shortcut, hidden])
        outputs = ELU(alpha=1.0)(hidden)

    pooling = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding=paddingType)(outputs)

    with tf.name_scope('layer4'):
        inputs = pooling
        features = 32
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType, kernel_regularizer=L2(regAmount),
                        kernel_initializer=initType)(inputs)
        hidden = BatchNorm(renorm=True)(hidden)
        # hidden = ELU(alpha=1.0)(hidden)
        # hidden = Conv3D(features, (3, 3, 3), padding=paddingType, kernel_regularizer=L2(regAmount),
        #                 kernel_initializer=initType)(hidden)
        # hidden = BatchNorm(renorm=True)(hidden)
        # shortcut = Conv3D(features, (1, 1, 1), strides=(1, 1, 1), padding=paddingType, kernel_initializer=initType)(
        #     inputs)
        # hidden = add([shortcut, hidden])
        outputs = ELU(alpha=1.0)(hidden)

    pooling = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding=paddingType)(outputs)


    with tf.name_scope('layer5'):
        inputs = pooling
        features = 16
        hidden = Conv3D(features, (1, 1, 1), padding='valid', kernel_regularizer=L2(regAmount),
                        kernel_initializer=initType)(inputs)
        hidden = BatchNorm(renorm=True)(hidden)
        # hidden = ELU(alpha=1.0)(hidden)
        # hidden = Conv3D(features, (3, 3, 3), padding=paddingType, kernel_regularizer=L2(regAmount),
        #                 kernel_initializer=initType)(hidden)
        # hidden = BatchNorm(renorm=True)(hidden)
        # shortcut = Conv3D(features, (1, 1, 1), strides=(1, 1, 1), padding=paddingType, kernel_initializer=initType)(
        #     inputs)
        # hidden = add([shortcut, hidden])
        outputs = ELU(alpha=1.0)(hidden)

    #pooling = AveragePooling3D(pool_size=(5, 6, 5), strides=(2, 2, 2), padding=paddingType)(outputs)
    pooling = AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding=paddingType)(outputs)

    ################# FC Block #############################
    hidden = Flatten()(pooling)

    # hidden = Dense(128, kernel_regularizer=L2(regAmount), kernel_initializer=initType, name='FullyConnectedLayer')(
    #     hidden)
    hidden = Dense(16, kernel_regularizer=L2(regAmount), kernel_initializer=initType, name='FullyConnectedLayer')(
        hidden)
    hidden = ELU(alpha=1.0)(hidden)
    hidden = Dropout(dropRate)(hidden)

    if includeGender:
        #scanner = Input((1,), name='Scanner')
        gender = Input((1,), name='Gender')
        hidden = concatenate([gender, hidden])

    prediction = Dense(1, kernel_regularizer=L2(regAmount), name='AgePrediction')(hidden)
    ########################################################################################

    if includeGender:
        model = Model(inputs=[t1Input, gender], outputs=prediction)
    else:
        model = Model(inputs=[t1Input], outputs=prediction)
    return model
