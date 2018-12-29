## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Dependencies
* Ubuntu 18.04
* CUDA 10.0
* Tensorflow 1.12.0 (Build from source)
* Numpy 1.15.4
* Matplotlib 3.0.2

### Directories Explanation
**Traffic_Sign_Classifier.ipynb** contains all the source code

**writeup_template.md** contains project description

**new_test_images** contains testing images from internet


### Dataset and Repository
The data set here is called German Traffic Signs (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

### Neural Network structure
| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5    	| 1x1 stride, Valid padding, outputs 28x28x32 	|
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, Valid padding, outputs 24x24x64   |
| RELU                  |                                               |
| Max Pooling	      	| 2x2 stride, outputs 16x16x64 				    |
| Convolution 3x3       | 1x1 stride, Valid padding, outputs 10x10x128  |
| RELU                  |                                               |
| Max Pooling           | 1x1 stride, outputs 5x5x128                   |
| Convolution 2x2       | 1x1 stride, Valid padding, outputs 4x4x256    |
| RELU                  |                                               |
| Max Pooling           | 2x2 stride, outputs 2x2x256                   |
| Flatten               | outputs 1024                                  |
| Fully connected       | outputs 512                                   |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully Connected       | outputs 256                                   |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully Connected       | outputs 43                                    |
| Softmax				| outputs 43     								|
