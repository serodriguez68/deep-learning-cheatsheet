# Part 1 - Data Pre-processing
# ----------------------------
# READ the 2-coding-a-cnn.md notes for more information about data Pre-Processing

# IMPORTANT NOTE: the dataset to run this example was not included in the
# repo because it is too big.  Please go to https://www.superdatascience.com/pages/deep-learning
# to download it.
TRAINING_SET_DIRECTORY = 'annotated_code/volume_1_supervised_deep_learning/part_2_convolutional_neural_network/dataset/training_set'
TEST_SET_DIRECTORY = 'annotated_code/volume_1_supervised_deep_learning/part_2_convolutional_neural_network/dataset/test_set'
INPUT_IMAGE_SIZE = (64, 64)
INPUT_IMAGE_SIZE_W_CHANNELS = INPUT_IMAGE_SIZE + (3,)
PIXEL_MAX_VALUE = 255

from keras.preprocessing.image import ImageDataGenerator

# This line combines Pixel Scaling and Image Augmentation
# train_datagen is a wrapper that wraps the training dataset and applies the describe operations when images
# are requested from it.
train_datagen = ImageDataGenerator(
        rescale=1./PIXEL_MAX_VALUE,  # Rescale all pixel values between 0 and 1
        shear_range=0.2,      # Random shear range (Image Augmentation)
        zoom_range=0.2,       # Random zoom range (Image Augmentation)
        horizontal_flip=True  # Random flipping enabled (Image Augmentation)
)

# The test-set will also be wrapped, but only the Pixel Scaling is applied
test_datagen = ImageDataGenerator(rescale=1./PIXEL_MAX_VALUE)

# Load dataset from prescribed folder structure using Keras' flow_from_directory method
# training_set is a lazy reference to the training dataset.
# The images are lazily given to the cnn in batches and the random transformations happen lazily as well.
# This is necessary because we cannot load all the images in memory at the same time.
training_set = train_datagen.flow_from_directory(
        TRAINING_SET_DIRECTORY,
        target_size=INPUT_IMAGE_SIZE,  # Resizing images to a standard shape
        batch_size=32,  # Size of the training batches in mini-batch gradient descent
        class_mode='binary'  # Type of problem
)
test_set = test_datagen.flow_from_directory(
        TEST_SET_DIRECTORY,
        target_size=INPUT_IMAGE_SIZE,
        batch_size=32,
        class_mode='binary'
)


# Part 2 - Make the CNN
# ----------------------------

# Import the libraries
from keras.models import Sequential  # Used to define an network container
from keras.layers import Convolution2D  # To deal with images (including color images)
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense  # Used for the Feed-Forward ANN after flattening has been done
from keras.layers import Dropout  # Overfitting control with dropout regularisation

# Defining the CNN using Sequential
image_classifier_cnn = Sequential()   # Creates an empty container for the network

# Add the First Convolution + ReLU Layer
image_classifier_cnn.add(
    Convolution2D(filters=32, kernel_size=(3, 3), strides=1, input_shape=INPUT_IMAGE_SIZE_W_CHANNELS, data_format='channels_last', activation='relu')
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

# Control overfitting with dropout
# In CNNs, dropout is typically used after the pooling layers, but this is a rough heuristic. It could also
# be used after the convolution layers.
# An alternative way of dropout in CNNS is to drop entire feature maps (as opposed to regions of each feature map).
# This is called 'SpatialDropout'
# image_classifier_cnn.add(Dropout(rate=0.3))

# We can keep adding convolutional-relu layers followed by max pooling layers in a similar way as above...
image_classifier_cnn.add(
    Convolution2D(filters=64, kernel_size=(3, 3), strides=1, data_format='channels_last', activation='relu')
    # We omit input_shape, because this time because Keras automatically figures out the input shape based on the
    # previous layer.  The input_shape parameter is only need for the first layer of the whole network
)
image_classifier_cnn.add(
    MaxPool2D(pool_size=2)
)

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
image_classifier_cnn.add(Dropout(rate=0.5))  # Overfitting control
# We use 1 node with a sigmoid function to make the network a classifier
image_classifier_cnn.add(
    Dense(units=1, activation='sigmoid')
)

# @see annotated_code/volume_1_supervised_deep_learning/part_1_artificial_neural_networks/ann_churn_classifier.py
#      for what each of this parameters mean
image_classifier_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 3 - Training the CNN on the images (and augmented images)
# ----------------------------
# TODO: you were going to build Tensorflow from source to get rid of the warning and speedup
# The warning: Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
# How to build: https://github.com/tensorflow/tensorflow/issues/8037#issuecomment-283831398

image_classifier_cnn.fit_generator(
        training_set,
        # steps_per_epoch: Number of images on the training set (without counting image augmentation).
        # We want all data to go through on each epoch.
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        # validation_steps: Number of images on the test set (without counting image augmentation).
        validation_steps=2000
)

# Part 4 - Save the Model for Future Use
# ----------------------------
saved_model_path = f'{SAVE_MODEL_DIRECTORY}/cnn_image_classifier.h5'
image_classifier_cnn.save(saved_model_path)
print('Saved to disk')


# ... later
# Part 5 - Load Model and Make Predictions
# ----------------------------
from keras.models import load_model
loaded_model = load_model(saved_model_path)
loaded_model.summary()

# Load images to make single predictions
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np


def load_image(img_path, target_size, max_pixel_value, show=False):
    img = image.load_img(img_path, target_size=target_size)
    img_tensor = image.img_to_array(img)             # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= max_pixel_value                    # Scale pixel value

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor

EXAMPLES_DIRECTORY = 'annotated_code/volume_1_supervised_deep_learning/part_2_convolutional_neural_network/dataset/single_prediction'
img1 = load_image(f'{EXAMPLES_DIRECTORY}/cat_or_dog_1.jpg', target_size=INPUT_IMAGE_SIZE, max_pixel_value=PIXEL_MAX_VALUE)
img2 = load_image(f'{EXAMPLES_DIRECTORY}/cat_or_dog_2.jpg', target_size=INPUT_IMAGE_SIZE, max_pixel_value=PIXEL_MAX_VALUE)

prediction1 = loaded_model.predict_classes(img1)
prediction2 = loaded_model.predict_classes(img2)

print(f'Image 1 has a predicted class of = {prediction1[0][0]}')
print(f'Image 2 has a predicted class of = {prediction2[0][0]}')
