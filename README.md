# Swarm-Research

## Training plan for image models(Without Swarm Optimisation)

- The models would be trained on the 2015 Kaggle Competition dataset-> [Link](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- The images in the dataset have an irregular size so we use the dataset -> [Link](https://www.kaggle.com/benjaminwarner/resized-2015-2019-blindness-detection-images)
- Total number of images is 35000 and we apply 5 fold stratified K Fold Validation.

- Models already covered
  - Resnet 50
  - Inception V3
  - Xception Net

## Deep Swarm(Swarm based optimisation for finding optimal Neural Network Architecture using Ant Colony Optimisation)

- Checked for compatibility of Code in TF 2.x using MNIST Pipeline
- Pipeline ready for training
- Hyper parameters chosen:
  - Image size -> 256
  - Number of Ants -> 8 or 16
  - Search Depth -> 8 or 16( Keeping four experiments in mind as of now. Permuting between ants and search depth)
  - Epochs -> 12 (Can be trained further)
- New set of experiments where results have been significantly better( AUC -> 0.9)

  - Image size -> 32(128 also on run, 256 seems to give Out of Memory error)
  - Number of Ants -> 8
  - Search Depth -> 32
  - Epochs -> 5(Cause the whole thing generalises quick)

- [Paper](https://arxiv.org/abs/1905.07350)

## psoCNN

- Implementation of [psoCNN](https://github.com/feferna/psoCNN) for APTOS 2015 dataset
- Script information:
  - Hyperparameters are defined in `main.py`
  - `particle.py` contains the definition of a particle (initialization and methods)
  - `population.py` contains the definition of a population (initialization and methods)
  - The algorithm is defined in `psoCNN.py`
  - Utility functions are defined in `utils.py`
- Hyperparameters used:
  - Image size -> 128
  - Number of particles in a population -> 10
  - Number of iterations -> 12
  - Number of algorithm runs -> 5
  - Epochs for training each particle -> 10
  - Epochs for training gBest -> 30
  - Range of number of layers for a model -> 10 - 16
- [Paper: Particle swarm optimization of deep neural networks architectures for image classification](https://www.sciencedirect.com/science/article/abs/pii/S2210650218309246?via%3Dihub)

```
@article {
  fernandes_junior_particle_2019,
	title = {Particle swarm optimization of deep neural networks architectures for image classification},
	volume = {49},
	issn = {22106502},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S2210650218309246},
	doi = {10.1016/j.swevo.2019.05.010},
	language = {en},
	urldate = {2019-07-06},
	journal = {Swarm and Evolutionary Computation},
	author = {Fernandes Junior, Francisco Erivaldo and Yen, Gary G.},
	month = sep,
	year = {2019},
	pages = {62--74},
}
```

### Running the algorithm

> ⚠️**The algorithm with the cuurent hyperparameters are very resource intrensive and were run on high performance hardware**

```bash
cd psoCNN
python main.py
```

> Install the [requirements](./requirements/tensorflow_requirements.txt) before running the script.

## LDWPSO CNN (Linearly Decreasing Particle Swarm Optimization Convolutional Neural Network)

- The program rewrites and uses part of the Hyperactive library
- Tested on MNIST
- [Paper](https://arxiv.org/abs/2001.05670)
