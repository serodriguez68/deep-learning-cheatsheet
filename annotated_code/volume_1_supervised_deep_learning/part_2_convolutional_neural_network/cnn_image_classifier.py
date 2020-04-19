# Part 1 - Data Preprocessing
# ----------------------------
# Loading images, loading labels and splitting the data:
# - This is mostly done by hand before we start coding.
# - Keras makes all the loading really simple for us if we provide all the data
#   following a folder structure that it expects.
# - See the "coding-a-cnn" notes for more details.

# Feature Scaling
# TODO: WHERE DOES FEATURE SCALING GOES? It was metioned in the video but not shown yet.

# IMPORTANT NOTE: the dataset to run this example was not included in the
# repo because it is too big.  Please go to https://www.superdatascience.com/pages/deep-learning
# to download it.

# Part 2 - Make the CNN
# ----------------------------

# Import the libraries
import keras
from keras.models import Sequential  # Used to define an network container
from keras.layers import Convolution2D  # To deal with images (including color images)
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense  # Used for the Feed-Forward ANN after flattening has been done

# Defining the CNN using Sequential
image_classifier_cnn = Sequential()   # Creates an empty container for the network

# Add the First Convolution + ReLU Layer
image_classifier_cnn.add(
    Convolution2D(filters=32, kernel_size=(3, 3), strides=1, input_shape=(64, 64, 3), data_format='channels_last', activation='relu')
    # filters: The number of feature detectors / filters. See notes for details on how many to put.
    # kernel_size: The size of each filter. If given an integer, it uses the same value in both directions.
    # strides: The stride to move the filters. When given an integer, it applies the same stride in both direction.
    #          Default (1, 1)
    # input_shape: Mandatory for the FIRST convolutional layer. Defines the size of the input images.
    #              e.g. (128, 128, 3) for 128 x 128 x 3 (RGB channels) images.
    #              If the images you are working with don't have a standard format, it is up to you to convert them
    #              into a standard shape during the data preprocessing step.
    # data_format: 'channels_last' indicates that in the `input_shape` argument has the channel number last.
    # activation: the activation function. Typically relu or leaky relu for convolutional layers.

)

# Add the First Max-Pooling Layer
image_classifier_cnn.add(
    MaxPool2D(pool_size=2)
    # pool_size: Size of the pooling window. Will halve the size of the feature map by 2.  2x2 is typically used.
    # strides: The stride of the pooling window. If not specified, defaults to pool_size.
)

#  We can keep adding convolutional-relu layers followed by max pooling layers in a similar way as above...

# Add the Flattening Layer
# Flattens the last max-pooling layer into a (very big) single vector
image_classifier_cnn.add(
    Flatten(data_format='channels_last')
    # data_format: @see data_format in the convolution layer
)

# Add the Full Connection Layer
# We will only add one hidden layer
image_classifier_cnn.add(
    Dense(units=128, activation='relu')
    # units: number of neurons in the layer. Which number to choose is an art.
    #        It is a trade-off between computation intensity + overfitting risk VS accuracy.
)
# We use 1 node with a sigmoid function to make the network a classifier
image_classifier_cnn.add(
    Dense(units=1, activation='sigmoid')
)

