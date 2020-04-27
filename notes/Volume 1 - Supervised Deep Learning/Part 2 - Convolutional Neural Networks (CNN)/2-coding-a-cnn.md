# Coding a CNN

## Tooling
We use the same tooling as for [coding an ANN][coding-an-ann-tooling]: (Theano OR Tensorflow) AND Keras.

## Data Pre-processing
Date pre-processing of images typically contains these steps:
1. Load the images and labels
1. Split the data into the _train_ and _test_ sets
1. Do __Image Scaling__
1. Do __Feature Scaling (Pixel Scaling)__
1. Do __Image Augmentation__

When using Keras, some of these steps are combined into single library calls, but it is important to understand
the motivation for each of each step separately.

See the [annotated code][cnn-code-detail] for more details.

### Loading the images and labels + Split the data into train and test sets 

Loading the images, loading the labels and splitting the data into train and test
is very easy to do in Keras if we organise our data following a prescribed folder structure. 
Keras calls this the [`flow_from_directory`][flow-from-directory-method] method.

The structure looks like this:
![Keras folder structure for image datasets][keras-folder-structure-for-image-datasets]
- Split all the data in 2 folders `test_set/` and `training_set/`.
- Separate images by category in subfolders e.g. `test_set/cats/` and `test_set/dogs/`.
- Name each file with the label e.g. `dog.1.jpg`...`dog.5000.jpg`.

### Image Scaling
The images in our training and test sets most likely have different sizes and aspect ratios. However, our
CNN takes a images of a fixed size as input, so we need to scale the images to the same `input size`.

There is an inherent trade-off with the _image input size_: 
- Bigger images give the CNN more information to work with, which can positively impact accuracy.
- However, bigger images also means a higher computational expense to train the network.

Both the training and test sets must be scaled equally.

### Feature Scaling (Pixel Scaling)
In addition to image scaling, we also re-scale the value of each pixel to take a value between `0` and `1`.
For example, if we receive an image whose pixel values range from `0` to `255` (e.g. an RGB image), we divide
each pixel by `255`.

Both the training and test sets must be scaled equally. 
 
### Image Augmentation

Image Augmentation is a technique that allows us to generate additional training images that are derived
from the original ones by randomly applying some simple transformations like rotations, zooming, shifting,
flipping, changes in brightness, etc.

![Example of Horizontal shift image augmentation][horizontal-shift-image-augmentation]
_Example of a horizontal shift image augmentation._
_Taken from [here.][image-augmentation-article]_

Image augmentation helps us in two ways:
1. Deep learning networks typically perform better when they are exposed to more data.
2. It exposes the network to plausible variations of the images that the model could
encounter during prediction.  For example, it is reasonable to think that someone may
take a picture with a parrot in the middle of the frame and some other person with the parrot on the
left.  Exposing the network to these variants during training makes it more robust to 
different scenarios. 

Image augmentation is only performed on the training set.

[Read this great article by Jason Brownlee if you want to know more][image-augmentation-article].

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

## Overfitting control using _dropout regularisation_

The motivation for using _dropout_ to control overfitting in CNNS  is largelythe same way as [for ANNs.][dropout-in-anns]

Implementation-wise, there are some nuances that only apply to CNNs:
- In CNNs, dropout is typically used after the pooling layers, but this is a rough heuristic.
It could also be used after the convolution layers.
- In CNNs,  an alternative form dropout is to drop entire feature maps (as opposed to regions of each feature map). 
This is called `SpatialDropout` and it is also supported in Keras.
- Dropout can also be applied to the fully connected hidden layers after flattening. This can be done in addition
to dropout in the convolution or max-pooling layers.


[Learn more about dropout regularisation here.][machine-learning-mastery-dropout]

[coding-an-ann-tooling]: ../Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/2-coding-an-ann.md#tooling
[keras-folder-structure-for-image-datasets]: ./keras-folder-structure-for-image-datasets.png
[image-augmentation-article]: https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
[cnn-code-detail]: ../../../annotated_code/volume_1_supervised_deep_learning/part_2_convolutional_neural_network/cnn_image_classifier.py
[flow-from-directory-method]: https://keras.io/preprocessing/image/#image-preprocessing
[horizontal-shift-image-augmentation]: ./horizontal-shift-image-augmentation.png
[dropout-in-anns]: ../Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/2-coding-an-ann.md#overfitting-control-using-_dropout-regularisation_
[machine-learning-mastery-dropout]: https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
