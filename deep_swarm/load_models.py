from deepswarm.backends import Dataset, TFKerasBackend
from deepswarm.deepswarm import DeepSwarm
from deepswarm.storage import Storage

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.compat.v1 import keras

"""
    Search for the latest date_time format in the saves directory. Move into
    the directory and check for best topology save and load models using keras
"""

model = keras.models.load_model("best-trained-topology")    
model.summary()