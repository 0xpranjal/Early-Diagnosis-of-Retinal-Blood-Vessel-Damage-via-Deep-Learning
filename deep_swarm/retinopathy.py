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

def format_images(arr):
    train_images = []
    for i in tqdm(arr):
        img =cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(128,128))
        train_images.append(img)
    x_train = np.asarray(train_images)
    return x_train


def preprocess_df():
    df = pd.read_csv("../labels/trainLabels15.csv")
    df['path'] = df['image'].apply(
        lambda x: "../resized_train_15/" + x + ".jpg")

    path = df['path'].values
    labels = df['level'].values

    x_train = format_images(path[:100])
    y_train = labels[:100]
    
    x_train = x_train.reshape(x_train.shape[0],128,128,3)
    x_val = format_images(path[100:125])
    y_val = labels[100:125]
    
    x_test = format_images(path[125:150])
    y_test = labels[125:150]
    x_test = x_test.reshape(x_test.shape[0],128,128,3) 
    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_df()
    dataset = Dataset(
        training_examples=x_train,
        training_labels=y_train,
        testing_examples=x_test,
        testing_labels=y_test,
        validation_split=0.2,
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
