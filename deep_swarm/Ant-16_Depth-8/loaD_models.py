from deepswarm.backends import Dataset, TFKerasBackend
from deepswarm.deepswarm import DeepSwarm
from deepswarm.storage import Storage

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.compat.v1 import keras

"""
    Search for the latest date_time format in the saves directory. Move into
    the directory and check for best topology save and load models using keras
"""
def macro_multilabel_auc(label, pred):
    aucs = []
    target_cols = [0, 1, 2, 3, 4]
    for i in range(len(target_cols)):
        aucs.append(roc_auc_score(label[:, i], pred[:, i]))
    return np.mean(aucs)


df = pd.read_csv("../dataset_folds/trainFolds15.csv")
y_test = df[df['fold'] == 4]
y_test =  y_test.iloc[:,3:].values


x_test = np.load("test_512.npy")

temp = []
for i in range(len(x_test)):
    temp.append(cv2.resize(x_test[i],(256,256)))

x_test = np.asarray(temp)
print(x_test.shape)
model = keras.models.load_model("4_Ant")    
#model.summary()


preds = model.predict(x_test)

roc = macro_multilabel_auc(y_test,preds)
print(f"ROC : {roc}")
print(f"Y Test Shape: {y_test.shape}")
print(f"Pred Shape :{preds.shape}")

#print(preds)
#print(preds.shape)
print("Done")


