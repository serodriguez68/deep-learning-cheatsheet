# Coding an LSTM

## Data Pre-Processing

As with any Neural Network, the data must be scaled prior to feeding it to the network. As usual, the scaler should
be fit on the __training data__ only and the same scaler must be applied at test time.

See [my feature scaling summary][feature-scaling-summary] for more general information about this topic.

- Hadelin de Ponteves recommends using `Normalization Scaling` when dealing with RNNs.
- He also says that  `Normalization Scaling` is especially important if the output layer of the RNN is a sigmoid.
- This is called the `MinMaxScaler` in `sklearn`

### Structure of the input data

- Keras's LSTM layer takes its inputs in a particular 3D shape that we must follow.  
- The structure is `(num_samples_on_this_training_batch, time_steps_in_batch, num_features_for_each_time_step)`.
- __At training time__: All samples of the same batch must have the same number of time steps, BUT different batches 
can have different `time_steps`.
   - After all, we are dealing with RNNs which by design can take sequences of varying lengths.
   - If we want to have training samples with different lengths on the same batch, then we must do padding.
- __At inference time__: the sequence can be of any length.
- `num_features_for_each_time_step` must be the same for all batches and for inference.
- `num_samples_on_this_training_batch` can change per batch.  This dimension of the input is automatically detected
by Keras's LSTM and therefore is not specified in the `input_shape` parameter (more on this below).
- If you want more details about how the different lengths work see [this stack overflow post][keras-lstm-with-different-sequence-lengths].
- This is an example of the structure of the input data using sentences of different lengths and word embeddings as
features per word.

 ![Keras LSTM input structure][keras-lstm-input-structure]


## Building the RNN
Keras makes it very easy with the `LSTM` layer. See [the annotated code][[lstm-stock-trend-sample-code]] for all the details.

## Arguments of the LSTM layer in Keras
The `LSTM` layer has 3 fundamental arguments.

### `units`
It is the number of neurons in each single-layer fully-connected neural network within the LSTM repeating group.

For example, if `units=250`, this means that the sigmoid neural layers and the tanh neural layers in the valves will
each have 250 neurons.

This also means that `units` determines the dimensionality of the output `ht` for each LSTM time-step. In the above
example, each LSTM time step will have an `ht` output of size 250x1.

`units` can be treated as a hyper-parameter, although it is common for it to be equal to the 
`num_features_for_each_time_step` 
- For example, if dealing with an NLP problem using embeddings of size 500, the `units` can be set to 500 as well. 


### `return_sequences`
By default, Keras' LSTM layer only returns the output of the last time step. If we want it to give an 
output per time step me must set `return_sequences=true`.

This is useful when connecting the output of one LSTM to as the input of another one (stacking LSTMS) or when we are
interested in the output per time step.

![keras lstm return sequence][keras-lstm-return-sequences-diagram]

### `input_shape`
- `input_shape = (time_steps_in_batch, num_features_for_each_time_step)`
- Keras will automatically detect the number of samples in the batch, so it doesn't need to be specified.
- If we have varying sequence lengths we can use `None` and Keras will accept batches with different lengths. For example
`input_shape=(None,500)` means varying sequence length and 500 features per time step.
- As with other layers in Keras, we only need to specify this for the first neural layer (regardless of it being LSTM or not). 
If you connect an LSTM to a preceding layer, this parameter will be inferred from the previous layer.
- At inference time, we can use any sequence length but we must respect the `num_features_for_each_time_step`

## Which Optimizer to use?
- The keras documentation recommends RMSprop as the optimizer for RNNs
- However, `adam` is also a safe bet (this is the one recommended by Hadelin for the Udemy RNN example)
- [Go here for more details about optimizers][ann-selecting-the-optimizer]

## Which loss function to use?
As with any neural network, it depends on the type of task you are doing. 
[See our summary about loss functions in ANNs for more information][ann-which-loss-function]
 
 
 ## Overfitting control using _dropout regularisation_
 
 Recommended read: Machine Learning Mastery has a great [article dedicated to this topic][mlm-lstm-dropout].
 
 LSTMs can easily overfit training data, so doing overfitting control is very important. _Dropout_ can be used as a 
 mechanism to reduce overfitting with basically the same motivation as [in ANNs.][dropout-in-anns].
 
 - Note that `Dropout` is not the only mechanism to control overfitting. There are other types of regularisation that 
 can be used but that are out of scope of this summary. 
 
 Implementation-wise, there are three types of `Dropout` in RNNs:
 - __Input Dropout:__ a random number of input features is dropped from the input to the RNN repeating group on each time step.
    - This is the `LSTM(droput=0.2)` argument in Keras.
 - __Recurrent Dropout:__ a random number of features from the input coming from the previous time step (`h_(t-1)`) is
 dropped in each time step.  Note that dropout is enforced at the point `h_(t-1)` gets into the repeating group at 
 time `t`, so it does NOT cause dropout in the output from the previous time step.
   - This is the `LSTM(recurrent_dropout=0.4)` argument in Keras.
 - __Output Dropout:__ a random number of features of the output from each time step is dropped.  This is typically used
 when the output of an RNN is going to be fed into some other neural network (maybe another LSTM).
   - This is done in Keras using the `Dropout(rate=0.2)` layer.
   

Rule of thumb: Hadelin from Udemy recommends starting with a dropout of 20% and explore from there.
RNNs require higher dropouts than ANNs or CNNs because they are very prone to overfitting, so a dropout
of 40% is not uncommon.
 
 
[feature-scaling-summary]: https://github.com/serodriguez68/machine-learning-cheatsheet/blob/master/cheatsheets/Part%201%20-%20Data%20Preprocessing/data-preprocessing.md#feature-scaling
[keras-lstm-return-sequences-diagram]: ./Keras-LSTM-return-sequences-diagram.png
[keras-lstm-with-different-sequence-lengths]:https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras
[keras-lstm-input-structure]: ./keras-lstm-input-structure.png
[lstm-stock-trend-sample-code]: ../../../annotated_code/volume_1_supervised_deep_learning/part_3_recurrent_neural_networks/lstm_stock_trend_regression.py
[ann-selecting-the-optimizer]: ../Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/2-coding-an-ann.md#selecting-the-optimizer
[ann-which-loss-function]: ../Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/2-coding-an-ann.md#which-loss-function-to-use
[dropout-in-anns]: ../Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/2-coding-an-ann.md#overfitting-control-using-_dropout-regularisation_
[mlm-lstm-dropout]: https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/

