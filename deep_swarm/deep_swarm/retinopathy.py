import numpy as np
import pandas as pd
import os
# !pip install deepswarm

import deepswarm
import tensorflow.compat.v1 as tf
from deepswarm.backends import Dataset, TFKerasBackend
from deepswarm.deepswarm import DeepSwarm
import cv2
from tqdm import tqdm

tf.disable_v2_behavior()

def format_images(arr,name):
    train_images = []
    for i in tqdm(arr):
        img =cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(128,128))
        train_images.append(img)
    x_train = np.asarray(train_images)
    np.save(f"{name}.npy",x_train)
    return x_train

def load_images(path):
    images = np.load(path)
    return images


def preprocess_df():
    df = pd.read_csv("../dataset_folds/trainFolds15.csv")
    df['path'] = df['image'].apply(
        lambda x: "../resized_train_15/" + x + ".jpg")
    df['path'] = df['image'].apply(lambda x: "../resized_train_15/" + x + ".jpg")
    fold_num = 4
    level = ['level_0','level_1','level_2','level_3','level_4']

    train =  df[df['fold'] != fold_num]
    test  =  df[df['fold'] == fold_num]
    """
    For Debug
    train = train.iloc[:100,:]
    test = test.iloc[200:300,:]
    
    """
    # train = load_images("train.npy")
    # test = load_images("test.npy")
    x_train = format_images(train['path'].values,"train")
    y_train = train[level].values
    x_train = x_train.reshape(x_train.shape[0],128,128,3)


    x_test = format_images(test['path'].values,"test")
    y_test = test[level].values
    x_test = x_test.reshape(x_test.shape[0],128,128,3)

    return x_train,y_train,x_test,y_test


if __name__ == '__main__':
    x_train, y_train,x_test, y_test = preprocess_df()
    dataset = Dataset(
        training_examples=x_train,
        training_labels=y_train,
        testing_examples=x_test,
        testing_labels=y_test,
        validation_split=0.3,
    )
    # Create backend responsible for training & validating
    backend = TFKerasBackend(dataset=dataset)
    # Create DeepSwarm object responsible for optimization
    deepswarm = DeepSwarm(backend=backend)
    # Find the topology for a given dataset
    topology = deepswarm.find_topology()
    # Evaluate discovered topology
    deepswarm.evaluate_topology(topology)
    # Train topology on augmented data for additional 50 epochs
    trained_topology = deepswarm.train_topology(topology, 5)
    # Evaluate the final topology
    deepswarm.evaluate_topology(trained_topology)
