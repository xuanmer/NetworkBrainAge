import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import r2_score,mean_absolute_error
from DataLoader import dataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K
import nibabel as nib

for NETWORK_NO in range(1, 8):
    print(f"\n===== Processing Subnetwork {NETWORK_NO} =====")
    
    # Load dataset
    data = pd.read_csv("zhe2/brain_2mm/zhe2_data_nerve_health.csv")
    
    # Load model for current subnetwork
    model = load_model(f'bty/brain-age/structural_brain-age/Models/Net{NETWORK_NO}')
    
    # Update file paths to point to subnetwork-specific inputs
    data['path'] = data['path'].str.replace('brain_resample.nii.gz', 
                                           f'brain_resample_subnet{NETWORK_NO}.nii.gz')
    
    # Generate predictions
    res = model.predict(dataGenerator([data.path.values, data.Gender.values],
                                      data.Age.values, 
                                      batch_size=1, 
                                      meanImg=None, 
                                      dim=(128,128,128),
                                      shuffle=False,
                                      augment=False,
                                      includeGender=False,
                                      normalise_mode=0,
                                      resize_img=False
                                     ), verbose=1, max_queue_size=32, workers=4, use_multiprocessing=False)
    
    # Process predictions
    predictions = res[:, 0]
    data['BA'] = predictions  # Brain Age
    data['PAD'] = data['BA'] - data['Age']  # Prediction Age Difference
    
    # Save results for current subnetwork
    output_file = f"zhe2/brain_2mm/results/zhe2_brainage_net{NETWORK_NO}.csv"
    data.to_csv(output_file)
    print(f"Results for subnetwork {NETWORK_NO} saved to: {output_file}")

print("\nAll subnetworks processed successfully!")
