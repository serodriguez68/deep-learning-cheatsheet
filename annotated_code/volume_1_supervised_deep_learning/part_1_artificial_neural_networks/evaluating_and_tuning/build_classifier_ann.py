import keras
from keras.models import Sequential  # To define the ANN
from keras.layers import Dense  # To build the hidden layers
from keras.layers import Dropout  # Dropout regularization to control overfitting


def build_classifier_ann(optimizer='adam'):
    classifier_ann = Sequential()
    classifier_ann.add(
        Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11)
    )
    # classifier_ann.add(Dropout(p=0.1))

    classifier_ann.add(
        Dense(units=6, kernel_initializer='uniform', activation='relu')
    )
    # classifier_ann.add(Dropout(p=0.1))

    classifier_ann.add(
        Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
    )
    classifier_ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier_ann
