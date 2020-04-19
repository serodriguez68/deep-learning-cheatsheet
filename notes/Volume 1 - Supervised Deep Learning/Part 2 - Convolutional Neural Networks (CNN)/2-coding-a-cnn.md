# Coding a CNN

## Tooling
We use the same tooling as for [coding an ANN][coding-an-ann-tooling]: (Theano OR Tensorflow) AND Keras.

## Data Pre-processing
### Dataset structure
For loading the images, loading the labels and splitting the data into train and test,
Keras makes it simple if we follow a prescribed folder structure. The structure looks like this:

![Keras folder structure for image datasets][keras-folder-structure-for-image-datasets]
- Split all the data in folders `test_set/` and `training_set`.
- Separate images by category in subfolders e.g. `test_set/cats/` and `test_set/dogs/`.
- Name each file with the label e.g. `dog.1.jpg`...`dog.5000.jpg`.

The bad news it that it is up to us to arrive to this folder structure however we see fit.

### Feature Scaling
Yes we need to apply feature scaling. 
TODO: this has been mentioned in the course but hast not been explained yet.

## Code
See the [annotated code][cnn-code-detail] for most of the information on how to actually do it.
The notes below just complement the annotated code.

### How many feature detectors / filters to put in each convolution layer?
This is a hyperparameter and selecting the value is an art and defined through experimentation.

A rule of thumb that is commonly used is doubling the filters on each convolutional layer. 
For example: 32 in the first convolutional layer, 64 in the second, 128 in the third... and so on.

### How many convolutional layers? 
Same thing. This is a hyperparameter that is found experimentally. Take into account that as the number of conv-layers
increase, so does the computation intensity of the task, so start with 1 or 3 and ramp up.

[coding-an-ann-tooling]: ../Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/2-coding-an-ann.md#tooling
[keras-folder-structure-for-image-datasets]: ./keras-folder-structure-for-image-datasets.png
[cnn-code-detail]: ../../../annotated_code/volume_1_supervised_deep_learning/part_2_convolutional_neural_network/cnn_image_classifier.py