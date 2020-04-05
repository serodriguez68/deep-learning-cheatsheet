# Part 1 - Data Preprocessing
# ----------------------------

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(
    './annotated_code/volume_1_supervised_deep_learning/part_1_artificial_neural_networks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  # Get rid of non-useful columns
y = dataset.iloc[:, 13].values

# One hot encode categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_gender = LabelEncoder()
X_enc = X.copy()
X_enc[:, 2] = labelencoder_gender.fit_transform(X[:, 2])
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_enc = np.array(ct.fit_transform(X_enc), dtype=np.float)
X_enc = X_enc[:, 1:]  # Drop one dummy variable to avoid dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=0)

# Feature Scaling
# This is absolutely compulsory for ANNs
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Make the ANN
# ----------------------------

# Import the libraries
import keras
from keras.models import Sequential  # To define the ANN
from keras.layers import Dense  # To build the hidden layers
from keras.layers import Dropout  # Overfitting control with dropout regularisation

# Defining the ANN
# There are 2 ways of defining an ANN in Keras:
#   Way 1. By defining the sequence of layers.
#   Way 2. By defining a graph.
# We show Way 1 here
classifier_ann = Sequential()  # Creates an empty ANN

# Add Input layer and the First Hidden Layer
#   `add` adds layers to the network
classifier_ann.add(
    Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11)
    # kernel_initializer: strategy for initializing the weights.
    # input_dim: Only needed for the fist hidden layer and defines the input layer.
    #            On following hidden_layers, Keras automatically detects the size of the input based on the
    #            previous layer.
)
classifier_ann.add(Dropout(p=0.1))
# p: fraction of the neurons in the layer that we want to randomly drop at each iteration.
#    Start with p=0.1 and see if that solves overfitting. If not, increase slowly until the problem is solved.

# Add a Second Hidden Layer
classifier_ann.add(
    Dense(units=6, kernel_initializer='uniform', activation='relu')
    # Note that input_dim parameter is not needed
)
classifier_ann.add(Dropout(p=0.1))

# Add the Output Layer
# We use 1 node with a sigmoid function to make the network a classifier
classifier_ann.add(
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
)

# Training the ANN
# `compile` allows us to configure the training.
classifier_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# optimizer: The algorithm used for optimization. 'adam' is a variant of stochastic gradient descent.
#            If we just provide the string, a default learning rate is used.
#            Alternatively we can give the `optimizer` argument an instance of the Adam optimizer with a custom
#            learning rate (see 2-coding-an-ann.md for more info).
# loss: Loss / cost function to use. We chose this based on the activation function of the output layer.
#       See notes for more info.
# metrics: List of metrics to report while your model is being trained.

# Fit the training set
classifier_ann.fit(X_train, y_train, batch_size=10, epochs=100)
# batch_size: number of observations before the loss function gets evaluated and the weights are updated.
# How to choose these parameters? It is an art. You need to do experimentation.


# Part 3 - Making the predictions and evaluating the model
# ----------------------------

# Predicting the test results
y_pred_probs = classifier_ann.predict(X_test)
y_pred = (y_pred_probs > 0.5)  # Convert probabilites to True or False

# Evaluating test 1set performance
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
print('Confusion Matrix')
print(cm)
print('Accuracy: ' + str(accuracy * 100) + ' %')

# Part 4 - Classifying a single observation in the future
# ----------------------------
# Observation:
# 'France', 600 credit score, male, 40 years, 3 yr tenure, balance 60,000, 2 products, true credit card
# true active member, estimated salary 50,000

# Get observation with categorical values encoded.
X_enc_homework = [[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]

# Scale data using the scaler fitted with training data
X_homework = sc.transform(X_enc_homework)

# Classify using ANN
y_homework_prob = classifier_ann.predict(X_homework)[0][0]
y_label = y_homework_prob > 0.5
print('The probability of exit is: ' + str(y_homework_prob) + ' (' + str(y_label) + ')')