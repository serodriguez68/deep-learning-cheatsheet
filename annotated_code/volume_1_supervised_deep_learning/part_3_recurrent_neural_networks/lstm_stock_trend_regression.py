# This file codes an LSTM designed to predict the "opening price"
# of the Google stock price, using the "Opening Price" from the previous days.
# This model is trained on 5 years of the Google Stock price.

# Run Configuration
# ----------------------------
# TODO

# Part 1 - Data Pre-processing
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv(
    './annotated_code/volume_1_supervised_deep_learning/part_3_recurrent_neural_networks/dataset/Google_Stock_Price_Train.csv'
)
# Select only the columns we care about and transform them to an numpy array
# Note the [['Open']] is to make sure that we end up with an numpy array of shape (1000, 1)
# and not a vector of shape (1000,)
training_set = dataset_train[['Open']].values

# == Feature Scaling
# As usual, we fit the feature scaler on the training set.
# Go to the notes on 2-coding-an-lstm.md for more details on why we use a normalization scaler (MinMaxScaler).
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))  # We want to scale to [0,1]
training_set_scaled = sc.fit_transform(training_set)

# == Create the training set
# Keras's LSTM layer takes its inputs in a particular 3D shape that we must follow.
# The structure is `(num_samples_on_this_training_batch, time_steps_in_batch, num_features_for_each_time_step)`.
# See the notes on "coding-an-lstm" for full details.

# For this example the training data will be made of:
# 1) X_train: The 60 previous data points for "Open price" to the day t (60 time steps per sample),
# 2) y_train: The "Open price" at the day t.
# How did we determine 60?  Experimentally. A small time window (like 1) leads to overfitting, a large time window
# (like 100) will lead to more a more computationally intensive calculation.
# In this particular problem 60 means use the information of 3 financial months to get the value of a particular day.
TIME_STEPS = 60

X_train = []
y_train = []

# We can only start at the 60th data point.
for i in range(TIME_STEPS, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - TIME_STEPS:i, 0])
    y_train.append(training_set_scaled[i, 0])

# Transform lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape training data to match Keras' LSTM input structure
# In this example, we have only one feature per time step. Reshape to make the 2D array 3D
X_train = np.reshape(X_train, X_train.shape + (1,))

# Part 2 - Building the LSTM
# ----------------------------
# For this example we are building a quite complex network of 4 LSTM layers stacked
# on top of each other.
# Note that this does't mean 4 time steps, in fact in this example each LSTM layer will have
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Add first LSTM layer with output dropout regularisation
regressor.add(
    # See the notes on "coding-an-lstm" for full details on each of this parameters.
    # units: Number of neurons in each single-layer fully-connected neural network within the LSTM repeating group.
    #        Also determines the size of the output of each time step. Mostly a hyper-parameter (see notes).
    #        Units don't have to match the number of features per time step necessarily. In this example we are
    #        increasing the dimensionality of the one feature to 50 by using 50 units.
    # return_sequences: By default, Keras' LSTM only returns the output of the last time step.
    #                   If we want one output vector per step, we need to set this to true.
    # input_shape: (time_steps_in_batch, num_features_for_each_time_step)
    #               (None, 500) means varying sequence length per batch and 500 features per time step.
    LSTM(units=50, return_sequences=True, input_shape=X_train.shape[1:3])
)

# This example uses Output Dropout regularisation.  RNNs have different types of dropout.
# See the notes on coding an LSTM for more details.
# A good starting point for LSTM dropout is 20%
regressor.add(Dropout(rate=0.2))

# Add second LSTM layer with dropout regularisation
regressor.add(
    # As with other layers in Keras, we only need to specify input_shape for the first neural layer
    # (regardless of it being LSTM or not).
    # If you connect an LSTM to a preceding layer, this parameter will be inferred from the previous layer.
    LSTM(units=50, return_sequences=True)
)
regressor.add(Dropout(rate=0.2))

# Add third LSTM layer with dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Add fourth LSTM layer with dropout regularisation
# Since in this regression example we only care about the output of the last time step, we
# let return_sequences=False (default)
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

# Add the output layer
# At this point the output of the Fourth LSTM layer is 1x50 (see units)
# We need a fully connected layer to map the output space of 5 to a single value
regressor.add(Dense(units=1))

# Compile the RNN
# The keras documentation recommends RMSprop as the optimizer for RNNs
# Hadelin from udemy says adam is usually a safe bet. He experimented with multiple optimizers
# and he found out that for this example adam worked better.
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Part 3 - Training the RNN
# ----------------------------
regressor.fit(X_train, y_train, epochs=100, batch_size=32)
# batch_size: number of observations before the loss function gets evaluated and the weights are updated.
# How to choose the number of epochs?
# - Experimentation
# - If the epochs number is too small, you will see that the RNN still hasn't converged
# - If the epochs number is too big, you risk overfitting


# Part 4 - Making the Predictions, and Visualizing the Results
# ----------------------------
# -- Massaging the test data to get it in the right formant

# Getting the test set (real stock price)
dataset_test = pd.read_csv(
    './annotated_code/volume_1_supervised_deep_learning/part_3_recurrent_neural_networks/dataset/Google_Stock_Price_Test.csv'
)
real_stock_price = dataset_test[['Open']].values  # The test set

# In this particular example, the first point in the test-set is the first financial day on 2017.
# For predicting that, we need the stock prices in the 60 previous financial_days. Those 60 previous days
# are the 60 last days of the training set.
# Don't worry about "using observed train data for testing" in this case; it's not what it looks like.
# Since we are dealing with a continuous time series, the particular exact combination of the last 60 days of the
# train set was never observed at training time. The closest training got to observe was
# (N_samples - 61) to (N_samples -1) as training input for the day at N_samples.
# The following lines basically augment the test data with the previous 60 days.
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
test_inputs = dataset_total[len(dataset_total) - len(dataset_test) - TIME_STEPS:].values.reshape(-1, 1)
test_inputs_scaled = sc.transform(test_inputs)  # Use the same scaling that was used at train time

# test_inputs_scaled is an 80x1 array that contains the sequence of the stock price, now we need to transform the
# data into the 3D shape Keras expects:
# (num_test_samples, time_steps_per_sample, num_features_for_each_time_step) => (20, 60, 1)
X_test = []
# We can only start at the 60th data point (the first point in the test set)
for i in range(TIME_STEPS, len(test_inputs_scaled)):
    X_test.append(test_inputs_scaled[i - TIME_STEPS:i, 0])
# Transform lists to numpy arrays
X_test = np.array(X_test)

# Reshape training data to match Keras' LSTM input structure
# In this example, we have only one feature per time step. Reshape to make the 2D array 3D
X_test = np.reshape(X_test, X_test.shape + (1,))

# -- Apply the regressor to the test data
y_predicted_scaled = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(
    y_predicted_scaled)  # predicted_stock_price contains the predicted google stock prices.

# -- Visualize the predicted stock price vs the actual stock price
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Real Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Financial Days from Jan 1 2017')
plt.ylabel('Stock Price USD')
plt.legend()
plt.show()

# Part 5 - Evaluating the Model
# ----------------------------
# We are going to show how to evaluate the regressor as if we cared about the accuracy of the predictions.
# However, note that in stock market predictions, we are typically interested in predicting the updwards / downwards
# trend of the stock rather than the actual value.
# As a result, measures lake the MSE are a less relevant measure o performance in these cases.
# For this particular example, Hadelin did a qualitative visual assessment of the model's performance using the graph.
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print('The RMSE is: ' + str(rmse))
