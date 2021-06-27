# Swarm-Research

## Training plan for image models
* Pretrain models on 2015 dataset and then train on 2019 dataset
* The number of epochs used is yet to be decided
* K-Fold validation on 2019 dataset.
    * Basically, the model would train as it is on 2015 dataset
    * On the 2019 dataset, folds would be used to check for accuracy 
* Total number of images is 35000 + 4000 -> 39000 images.(Cause we got V100, can change this as well)
