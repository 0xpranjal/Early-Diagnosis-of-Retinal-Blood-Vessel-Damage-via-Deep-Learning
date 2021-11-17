
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.compat.v1 import keras
# from tensorflow.compat.v1. keras.datasets import mnist
# from tensorflow.compat.v1.keras.datasets import fashion_mnist
# from tensorflow.compat.v1.keras.datasets import cifar10
from tensorflow.compat.v1.keras import backend
#
# from sklearn.model_selection import train_test_split

from population import Population

import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image

from copy import deepcopy

def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def preprocess_image(image_path, desired_size=32):
    im = Image.open(image_path)
    im = im.resize((desired_size,desired_size), resample=Image.LANCZOS)
    
    return im


class psoCNN:
    def __init__(self, n_iter, pop_size, batch_size, epochs, min_layer, max_layer, \
        conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, dropout_rate):
        
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.epochs = epochs

        self.batch_size = batch_size
        self.gBest_acc = np.zeros(n_iter)
        self.gBest_test_acc = np.zeros(n_iter)

        input_width = 128
        input_height = 128
        input_channels = 3
        output_dim = 5
        self.fold = 4
        self.img_size = 128


        #reading the data info
        df = pd.read_csv('../dataset_folds/trainFolds19_without_OH.csv')
        train_df = df[df['fold'] != self.fold]
        test_df = df[df['fold'] == self.fold]
        N = train_df.shape[0]
        x_train = np.empty((N, self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Resize and resample train images
        for i, image_id in enumerate(tqdm(train_df['id_code'])):
            x_train[i, :, :, :] = preprocess_image(
                f'../resized_train_19/{image_id}.jpg',
                desired_size = self.img_size
            )
        
        # Set the labels as diagnosis column in dataset
        y_train = pd.get_dummies(train_df['diagnosis']).values

        Nt = test_df.shape[0]
        x_test = np.empty((Nt, self.img_size, self.img_size, 3), dtype=np.uint8)

        # Resize and resample test images
        for i, image_id in enumerate(tqdm(test_df['id_code'])):
            x_test[i, :, :, :] = preprocess_image(
                f'../resized_train_19/{image_id}.jpg',
                desired_size = self.img_size
            )

        y_test = pd.get_dummies(test_df['diagnosis']).values

        print("X Test Shape: ", x_test.shape)
        print("Y Test Shape: ", y_test.shape)

        # Initialize population
        print("Initializing population...")
        self.population = Population(pop_size, min_layer, max_layer, input_width, input_height, input_channels, conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim)
        
        #setting the first particle as gBest
        print("Verifying accuracy of the current gBest...")
        print(self.population.particle[0])
        self.gBest = deepcopy(self.population.particle[0])
        self.gBest.model_compile(dropout_rate)

        hist = self.gBest.model_fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        test_metrics = self.gBest.model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
        self.gBest.model_delete()
        print(hist.history.keys())
        self.gBest_acc[0] = hist.history['acc'][-1]
        self.gBest_test_acc[0] = test_metrics[1]
        
        self.population.particle[0].acc = hist.history['acc'][-1]
        self.population.particle[0].pBest.acc = hist.history['acc'][-1]

        print("Current gBest acc: " + str(self.gBest_acc[0]) + "\n")
        print("Current gBest test acc: " + str(self.gBest_test_acc[0]) + "\n")

        # Searching for new gBest in the population
        print("Looking for a new gBest in the population...")
        for i in range(1, self.pop_size):
            print('Initialization - Particle: ' + str(i+1))
            print(self.population.particle[i])

            self.population.particle[i].model_compile(dropout_rate)
            hist = self.population.particle[i].model_fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
            self.population.particle[i].model_delete()
            self.population.particle[i].acc = hist.history['acc'][-1]
            self.population.particle[i].pBest.acc = hist.history['acc'][-1]
            
            #Updating the gBest if a particle with better accuracy is found
            if self.population.particle[i].pBest.acc >= self.gBest_acc[0]:
                print("Found a new gBest.")
                self.gBest = deepcopy(self.population.particle[i])
                self.gBest_acc[0] = self.population.particle[i].pBest.acc
                print("New gBest acc: " + str(self.gBest_acc[0]))
                
                self.gBest.model_compile(dropout_rate)
                test_metrics = self.gBest.model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
                self.gBest_test_acc[0] = test_metrics[1]
                print("New gBest test acc: " + str(self.gBest_acc[0]))
            
            self.gBest.model_delete()


    def fit(self, Cg, dropout_rate):
        
        df = pd.read_csv('../dataset_folds/trainFolds19_without_OH.csv')
        train_df = df[df['fold'] != self.fold]
        test_df = df[df['fold'] == self.fold]
        print("bef N: ", train_df.shape[0])
        N = train_df.shape[0]
        x_train = np.empty((N, self.img_size, self.img_size, 3), dtype=np.uint8)

        # Resize and resample images
        for i, image_id in enumerate(tqdm(train_df['id_code'])):
            x_train[i, :, :, :] = preprocess_image(
                f'../resized_train_19/{image_id}.jpg',
                desired_size = self.img_size
            )
        
        y_train = pd.get_dummies(train_df['diagnosis']).values

        Nt = test_df.shape[0]
        x_test = np.empty((Nt, self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Resize and resample images
        for i, image_id in enumerate(tqdm(test_df['id_code'])):
            x_test[i, :, :, :] = preprocess_image(
                f'../resized_train_19/{image_id}.jpg',
                desired_size = self.img_size
            )

        y_test = pd.get_dummies(test_df['diagnosis']).values

        for i in range(1, self.n_iter):            
            gBest_acc = self.gBest_acc[i-1]
            gBest_test_acc = self.gBest_test_acc[i-1]

            for j in range(self.pop_size):
                print('Iteration: ' + str(i) + ' - Particle: ' + str(j+1))

                # Update particle velocity
                self.population.particle[j].velocity(self.gBest.layers, Cg)

                # Update particle architecture
                self.population.particle[j].update()

                print('Particle NEW architecture: ')
                print(self.population.particle[j])

                # Compute the acc in the updated particle
                self.population.particle[j].model_compile(dropout_rate)
                hist = self.population.particle[j].model_fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
                self.population.particle[j].model_delete()

                self.population.particle[j].acc = hist.history['acc'][-1]
                
                f_test = self.population.particle[j].acc
                pBest_acc = self.population.particle[j].pBest.acc

                if f_test >= pBest_acc:
                    print("Past pBest acc: " + str(pBest_acc))
                    pBest_acc = f_test
                    self.population.particle[j].pBest = deepcopy(self.population.particle[j])

                    if pBest_acc >= gBest_acc:
                        print("Found a new gBest.")
                        gBest_acc = pBest_acc
                        self.gBest = deepcopy(self.population.particle[j])
                        
                        self.gBest.model_compile(dropout_rate)
                        hist = self.gBest.model_fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
                        test_metrics = self.gBest.model.evaluate(x=x_test, y=y_test, batch_size=self.batch_size)
                        self.gBest.model_delete()
                        gBest_test_acc = test_metrics[1]

                
            self.gBest_acc[i] = gBest_acc
            self.gBest_test_acc[i] = gBest_test_acc

            print("Current gBest acc: " + str(self.gBest_acc[i]))
            print("Current gBest test acc: " + str(self.gBest_test_acc[i]))

    def fit_gBest(self, batch_size, epochs, dropout_rate):

        df = pd.read_csv('../dataset_folds/trainFolds19_without_OH.csv')
        train_df = df[df['fold'] != self.fold]
        test_df = df[df['fold'] == self.fold]
        print("bef N: ", train_df.shape[0])
        N = train_df.shape[0]
        x_train = np.empty((N, self.img_size, self.img_size, 3), dtype=np.uint8)

        # Resize and resample images
        for i, image_id in enumerate(tqdm(train_df['id_code'])):
            x_train[i, :, :, :] = preprocess_image(
                f'../resized_train_19/{image_id}.jpg',
                desired_size = self.img_size
            )
        
        y_train = pd.get_dummies(train_df['diagnosis']).values

        print("\nFurther training gBest model...")
        self.gBest.model_compile(dropout_rate)

        trainable_count = 0
        for i in range(len(self.gBest.model.trainable_weights)):
            trainable_count += backend.count_params(self.gBest.model.trainable_weights[i])
            
        print("gBest's number of trainable parameters: " + str(trainable_count))
        self.gBest.model_fit_complete(x_train, y_train, batch_size=batch_size, epochs=epochs)

        return trainable_count
    
    def evaluate_gBest(self, batch_size):

        df = pd.read_csv('../dataset_folds/trainFolds19_without_OH.csv')
        train_df = df[df['fold'] != self.fold]
        test_df = df[df['fold'] == self.fold]
        
        Nt = test_df.shape[0]
        x_test = np.empty((Nt, self.img_size, self.img_size, 3), dtype=np.uint8)

        for i, image_id in enumerate(tqdm(test_df['id_code'])):
            x_test[i, :, :, :] = preprocess_image(
                f'../resized_train_19/{image_id}.jpg',
                desired_size = self.img_size
            )

        y_test = pd.get_dummies(test_df['diagnosis']).values

        print("\nEvaluating gBest model on the test set...")
        
        metrics = self.gBest.model.evaluate(x=x_test, y=y_test, batch_size=batch_size)

        print("\ngBest model loss in the test set: " + str(metrics[0]) + " - Test set accuracy: " + str(metrics[1]))
        return metrics
