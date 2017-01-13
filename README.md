# Generative Models of Neuron Morphology
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/RoozbehFarhoodi/McNeuron/blob/master/LICENCE) 

The goals is to make a generative model for the neurons. The input of each algorithm is a database of the neurons and the output is the generated neurons that looks like the same in morphology. 
Here we pursuit two approaches:

## Markov chain Monte Carlo:
In MCMC, a random sample of distribuion is drown randomly by perturbting in each iteration. More especifically, it each iteration, the current state is changed to generate a proposal sample and if the probability of the proposal is higher, MCMC jumps to propsal, otherwise with the ratio of these two probabilities it may jump. 
Here, we look at different features of the neuron like:

 - `Number of branching point`
 - `Histogram of distance from soma`
 - `Histogram of diameters`
 - ...

and make a probability space with these features and then drow a sample from it by MCMC.

## GAN (Generative Adversarial Network):
The basic idea of generative adversarail model [[Goodfellow et al]](https://arxiv.org/pdf/1406.2661v1.pdf) is to learn two models, one for generating and one for discrimination. Generative model explore in the space of shape generates variaty of images and then discriminative model tries to differentiate the generated image from real images. If the network learns well, the generative model make the images very close to the data such that discriminative model can only reject by chance.
The images here are the neuron nd the gray level revea the depth of each compartment of the neuron.
The script is stolen from [here](https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py). Originally it was used for generating digit from MNIST.
### Installation

Clone the repository.

```bash
$ git clone https://github.com/RoozbehFarhoodi/McNeuron
```
## Dependencies

- [numpy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scipy](https://www.scipy.org/)
- [keras](https://keras.io/)

## Members

- [Roozbeh Farhoodi](http://kordinglab.com/people/roozbeh_farhoodi/index.html), Sharif University
- [Pavan Ramkumar](http://kordinglab.com/people/pavan_ramkumar/index.html), Northwestern University
- [Hugo Fernandes](http://kordinglab.com/people/hugo_fernandes/index.html), Northwestern University

## Acknowledgement

- [Konrad Koerding](http://kordinglab.com)
- [COGC](http://cogc.ir/) for funding
- Package is developed in [Konrad Kording's Lab](http://kordinglab.com/) at Northwestern University

## License

MIT License Copyright (c) 2016 Roozbeh Farhoodi
