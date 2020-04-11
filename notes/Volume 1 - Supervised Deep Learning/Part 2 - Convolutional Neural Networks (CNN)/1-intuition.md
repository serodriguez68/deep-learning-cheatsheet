# Convolutional Neural Network (CNN) Intuition

## What are Convolutional Neural Networks Used for?

CNNs are typically used for image recognition tasks like image classification, but they also have applications
in recommender systems, natural language processing and time series.

In this work, we will anchor all the explanation of CNNs through image classification tasks, but the knowledge is
transferable to other applications.
  
## Input Structure
CNNs work with images / video-frames as inputs. This might look complicated, but under the hood images are just
N-dimensional arrays. CNNs work with these arrays as input.

For example, a Black and White image is a 2D array in which each cell contains a number between 0 and 255, indicating 
the "level of white" of the pixel.

A color image is a 3D array, with layers for red, green and blue (RGB). 

![CNN Input Structure](cnn-input-structure.png)

The next example shows how a "smiley face" can be represented using a simple _Black (1) OR White (0)_ encoding.
![CNN Sample Input](cnn-sample-input-face.png)

## Structure of a CNN (Steps)
### 1. Convolution Operation
#### ReLU Layer
### 2. Pooling
### 3. Flattening
### 4. Full Connection

## Softmax & Cross-Entropy


## Complementary Readings
- https://www.superdatascience.com/blogs/the-ultimate-guide-to-convolutional-neural-networks-cnn