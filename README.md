# Efficient Architecture Search in Multidimensional Space using Swarm Intelligence

## Deep CNN Models trained(Imagenet based models)

  - Resnet 50
  - Inception V3
  - Xception Net

## Deep Swarm(Swarm based optimisation for finding optimal Neural Network Architecture using Ant Colony Optimisation)

- Hyper parameters Used:
  - Image size -> 32
  - Number of Ants -> 8 or 16
  - Search Depth -> 8 or 16
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

