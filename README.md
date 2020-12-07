# George Washington Neural Network
This repository contains a neural network library built from scratch for use in Neural Network courses at
The George Washington University. It is a basic implementation that provides a number of layers, activation
functions, and loss functions for students to explore how networks work at a low level.

## Convolutional Neural Network (CNN) Functionality
The branch "convolutional-neuralnet-so" is an extension of the GWU neural network library with the added cnn functionality implemented by [@shivaomrani](https://github.com/shivaomrani). The neural network functionality is tested on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. You can run either jupyter notebook or a python file to test the CNN. Jupyter notebook is advised for more details. Run the the following command to test CNN from python file:

```
python cnn.py
```
Run the the following command to test CNN from jupyter notebook, and then select cnn.ipynb from the interface:

```
jupyter notebook
```


### General Tips
Here are the current configurations of "cnn.py" that result in an accuracy of 98%:

1) 1 Conv2D layer: 3x3 kernel, 1x1 stride, num_filters = 2 (default if not specified)

2) 1 MaxPooling2D layer: 2x2 pool size, 2x2 stride (equal to pool size if not specified)

3) 1 Flatten layer

4) For faster training speed, please select smaller x train and y train datasets. Currently, they are set to 4,000 samples. You can change this number in line 35 of "cnn.py".

5) You can specify the 2 classes for binary classification by specifying them in line 35 and 36 (currently, the classification is performed for digits 3 and 7)

### My added functionalities
The core work of my project was adding 3 distinct layers along with their forward and backward logic to "gwu_nn/layers.py" file:

1) Conv2D

2) MaxPooling2D

3) Flatten
