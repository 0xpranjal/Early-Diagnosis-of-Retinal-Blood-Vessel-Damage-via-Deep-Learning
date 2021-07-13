import numpy as np
import pandas as pd

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from tensorflow.python.framework.config import set_memory_growth


import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras as k
tf.disable_v2_behavior()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



from tensorflow.compat.v1.keras import models, layers
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.losses import CategoricalCrossentropy
import cv2, json

# ignoring warnings
import warnings
warnings.simplefilter("ignore")

IMG_SIZE = 256
WORK_DIR = "../"
MODEL_PATH = "epoch1"
FOLD = 2


def load_data():
    data = pd.read_csv("../dataset_folds/trainFolds15_without_OH.csv")
    data['image_id'] = data['image'] + ".jpg"
    data = data[['image_id', 'level','fold']]
    data.columns = ['image_id', 'label','fold']
    data.label = data.label.astype("str")
    #data = data.sample(200)
    return data

def getDataGen(train_data,valid_data):
    train_generator = ImageDataGenerator().flow_from_dataframe(
        train_data,
        directory=WORK_DIR + "resized_train_15/",
        x_col="image_id",
        y_col="label",
        # weight_col = None,
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="categorical",
        classes = ["0","1","2","3","4"],
        batch_size=32,
        subset="training",
    )

    valid_generator = ImageDataGenerator(validation_split=0.99).flow_from_dataframe( # Already split for folds
        valid_data,
        directory=WORK_DIR + "resized_train_15/",
        x_col="image_id",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="categorical",
        classes = ["0","1","2","3","4"],
        batch_size=32,
        subset="validation"
    )

    return train_generator,valid_generator

def load_model(path):
    model = k.models.load_model(path)
    return model

if __name__ =='__main__':
    df = load_data()
    train_data = df[ df['fold'] != FOLD]
    valid_data = df[ df['fold'] == FOLD]

    train_generator,valid_generator = getDataGen(train_data, valid_data)
    model_check = ModelCheckpoint(
        "./mod_name.h5",
        monitor="val_auc",
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode="max"
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=7,
        verbose=1,
        mode="min",
        restore_best_weights=False
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=2,
        verbose=1,
        mode="min",
        min_delta=0.0001,
    )
    model = load_model(MODEL_PATH)
    model.compile(optimizer="adam",
                  loss=CategoricalCrossentropy(label_smoothing=0.3, reduction="auto", name="categorical_crossentropy"),
                  metrics=[tf.keras.metrics.AUC()]
                  )

    history = model.fit(
        train_generator,
        epochs=2,
        validation_data=valid_generator,
        callbacks=[model_check, early_stop, reduce_lr])
    
    hist_df = pd.DataFrame(history.history)
    json_name = 'history.json'
    with open( json_name,mode ='w') as f:
        hist_df.to_json(f)


