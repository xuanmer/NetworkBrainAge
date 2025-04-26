import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.optimizer_v2.adam import Adam
from DataLoader import dataGenerator
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from SFCN import generateAgePredictionSFCNet

# For each Yeo-7 network, we trained brain age model using an improved SFCN model31 on the training set of the UK Biobank

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.per_process_gpu_memory_fraction = 0.9
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

for i in range(7):
    if i not in [3] and os.path.exists("./Models/Net"+str(i+1)):
        print("exists BrainAgeSFCN_Net"+str(i+1))
        continue
    data=pd.read_csv("./data_nerve_health_new.csv")
    data['path']=data['path'].str.replace('freesurfer/Subjects','seven_networks/network'+str(i+1))
    data['path']=data['path'].str.replace('brain_2mm.nii.gz','net.nii.gz')
    train,test=train_test_split(data, test_size = 0.4,random_state=45346)
    val,test=train_test_split(test, test_size = 0.5,random_state=257572)
    nEpochs=150
    batchSize = 32
    steps_per_epoch= train.shape[0]//batchSize
    validation_steps = val.shape[0]//batchSize
    dataShape=(128,128,128)
    default_parameters = [0.01,1e-6,None,'IncludeGender',0.00005,0.2,40,10]
    learning_rate, decayRate, meanImg, gender,regAmount, dropRate, maxAngle,maxShift = default_parameters
    if gender == 'RandomInput':
        gender_train = np.random.rand(train.Gender.shape[0])
        gender_val = np.random.rand(val.Gender.shape[0])
    else:
        gender_train = train.Gender.values
        gender_val = val.Gender.values

    model = load_model("./Models/Net"+str(i))
    learning_rate=0.001
    #model=generateAgePredictionSFCNet(dataShape,regAmount=regAmount,dropRate=dropRate,includeGender=False)
    adam = Adam(learning_rate=learning_rate, decay=decayRate)
    model.compile(loss='mean_absolute_error',optimizer=adam, metrics=['mae','mse'])
    mc = ModelCheckpoint('./Models/Net'+str(i+1),verbose=1,mode='min',save_best_only=True)
    early = EarlyStopping(patience=90, verbose=1)
    reduce_lr_plateau=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=0, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)
    h = model.fit(dataGenerator([train.path.values,gender_train],train.Age.values, batch_size = batchSize, meanImg=meanImg,dim=dataShape,shuffle=True,augment=False,maxAngle=maxAngle,maxShift=maxShift,includeGender=False,resize_img=False),
                            validation_data=dataGenerator([val.path.values,gender_val],val.Age.values, batch_size = batchSize, meanImg=meanImg,dim=dataShape,shuffle=False,augment=False,includeGender=False,resize_img=False),
                            validation_steps=validation_steps,
                            steps_per_epoch=steps_per_epoch,
                            epochs=nEpochs,
                            verbose=1,
                            max_queue_size=32,
                            workers=1,
                            use_multiprocessing=False,
                            callbacks=[mc,early,reduce_lr_plateau]
                            )

    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('SFCNet Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig("SFCNet Loss"+str(i+1)+".png")
    #plt.show()