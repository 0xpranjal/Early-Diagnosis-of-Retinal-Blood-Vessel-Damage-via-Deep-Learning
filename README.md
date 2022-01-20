# Efficient Architecture Search in Multidimensional Space using Swarm Intelligence
by [Pranjal Bhardwaj](https://www.linkedin.com/in/pranjal-bhardwaj-a85263188/), [Thejineaswar Guhan](https://www.linkedin.com/in/gthejin/), [Prajjwal Gupta](https://www.linkedin.com/in/prajjwal-gupta-9bb9381a5/), [Kathiravan Srinivasan](https://scholar.google.com/citations?user=pY3jLUkAAAAJ&hl=en), Senior Member, IEEE, and [Chuan-Yu Chang](https://scholar.google.com/citations?user=4iL0d3kAAAAJ&hl=en), Senior Member, IEEE

## Abstract
In this work, we propose the usage of tailor fitted models proposed by Swarm Algorithms to detect diabetic retinopathy in fundus images. We used two different Swarm Algorithm, Ant Colony Optimisation and Particle Swarm Optimisation. For the same, we use Deepswarm and psoCNN implementation to automatically search for deep Convolutional Neural Networks. We examined performance of the two architectures with respect to ImageNet models(Xception, Resnet 50, and InceptionV3). The public dataset APTOS 2019 was used for the training and evaluation of these models. With particle swarm optimization our models combinedly achieve an AUC ROC of 0.98, Accuracy of 89.9%, and Quadratic weighted Cohen Kappa score of 0.913. We observe that these optimization techniques tailor make models for a dataset which has the potential to outperform with ImageNet models.


## Deep CNN Models trained(Imagenet based models)

  - Resnet 50
  - Inception V3
  - Xception Net

## Ant Colony Optimisation
- Uses [Deep Swarm](https://github.com/Pattio/DeepSwarm) for the implementation.
- Hyper parameters Used:
  - Image size -> 32
  - Number of Ants -> 8 or 16
  - Search Depth -> 8 or 16
- [Paper -> DeepSwarm: Optimising Convolutional Neural Networks using Swarm Intelligence](https://arxiv.org/abs/1905.07350)

## Particle Swarm Optimisation

- Uses [psoCNN](https://github.com/feferna/psoCNN) for APTOS 2015 dataset
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

