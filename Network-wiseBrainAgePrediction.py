'''
Apply to unseen data (the UK Biobank training set)
'''
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error
from DataLoader import dataGenerator
from sklearn.model_selection import train_test_split

data=pd.read_csv("./data_nerve_health.csv")

NETWORK_NO=7
model=load_model('./Models/Net'+str(NETWORK_NO))
data['path']=data['path'].str.replace('freesurfer/Subjects','seven_networks/network'+str(NETWORK_NO))
data['path']=data['path'].str.replace('brain_2mm.nii.gz','subnet.nii.gz')
train,test=train_test_split(data, test_size = 0.4,random_state=45346)
val,test=train_test_split(test, test_size = 0.5,random_state=257572)
res=model.predict(dataGenerator([test.path.values,test.Gender.values],test.Age.values,batch_size=1,meanImg=None,dim=(128,128,128),shuffle=False,augment=False,includeGender=False,normalise_mode=0,resize_img=False),
              verbose=1,
              max_queue_size=32,
              workers=4,
              use_multiprocessing=False,
              )
predictions=res[:,0]
test['BA']=predictions
y=test.Age.values
print('R^2: ',r2_score(y,predictions))
print('MAE: ',mean_absolute_error(y,predictions))
plt.scatter(y,predictions,label='Predictions')
y_range = np.arange(20,np.max(y))
plt.plot(y_range,y_range,c='black',ls='dashed',label='45 deg line')
plt.xlabel('Age')
plt.ylabel('Predicted Age')
plt.title('Prediction UK Biobank')
plt.legend()
plt.show()
test.to_csv("./data_network"+str(NETWORK_NO)+"_nerve_health_BA.csv")
brainAge = pd.DataFrame({'PredictedBrainAge':np.array(predictions)},index = test['id'])
brainAge.to_csv('./results/pred.txt')