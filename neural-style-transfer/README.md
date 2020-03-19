# Neural style transfer

## Overview

For this deep learning project, we have decided to try to tackle the task of neural style transfer.

The goal of such a task is to combine the content and style of two different images.

## Context

Project realized as part of the *Deep learning* course given by Professor **Louppe** to the master students of Civil Engineering at the [University of Liège](https://www.uliege.be/) during the academic year 2019-2020.

The project has been realised by :

* Maxime Meurisse
* Adrien Schoffeniels
* Valentin Vermeylen

## File organisation

The archive consists of several folders :

* `python/` : contains the python implementation of our neural network;
* `resources/` : contains the resources used for the project (the images);
* `results/` : contains the results produced by our neural network (images and graphs).

### Code organisation

The code consists of several files :

* `models/` : contains the implementations of the tested and used models;
* `chart.py` : library allowing to make graphs;
* `image.py` : contains all functions for processing images;
* `loss.py` : contains the implementation of our loss functions (content and style);
* `main.py` : main file of the project;
* `process.py` : contains the methods for using the network and loss functions to produce the output image.

## How to use

First of all, you have to create the right environment with anaconda.

```
conda env create -f environment.yml
```

The only elements to be modified are the global variables in the `main.py` file.

```
STYLE_PATH = '../resources/style/'
CONTENT_PATH = '../resources/content/'
RESULT_PATH = '../results/'

STYLE_IMG = 'starry-night.png'
CONTENT_IMG = 'eiffel-tower.png'
RESULT_IMG = 'starry-night-eiffel-tower.png'

NUM_STEPS = 300
STYLE_WEIGHT = 1000000
CONTENT_WEIGHT = 10
```

Then simply call the main file.

```
python main.py
```

## Inspiration sources

* [PyTorch tutorial - Neural style transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
* [VGG19 architecture - Kaggle](https://www.kaggle.com/keras/vgg19)
