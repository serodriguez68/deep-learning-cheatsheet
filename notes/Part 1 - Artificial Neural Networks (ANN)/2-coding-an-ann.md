# Coding an ANN

## Tooling
- __Theano__ and __Tensorflow__: Open-source tensor numerical computation libraries based on numpy. Very efficient. Can run on the CPU and GPU.
  - They are mostly used for research in the deep learning field. They require many lines of code to develop a network from scratch.
- __Keras__: Wraps Theano and Tensorflow and is used mostly to build industrial models, not research.
  - Requires a few lines of code to build very complex deep learning models.
  - You can use Keras with Tensorflow OR Theano as a backend. It uses Tensorflow by default.

## Feature Scaling
Feature scaling is absolutely compulsory for ANNs, 
see [my summary on feature scaling](https://github.com/serodriguez68/machine-learning-cheatsheet/blob/master/cheatsheets/Part%201%20-%20Data%20Preprocessing/data-preprocessing.md#feature-scaling)
for more information.

## Code
[This annotated code](./../../annotated_code/volume_1_supervised_deep_learning/part_1_artificial_neural_networks/ann_churn_classifier.py)
shows a the step-by-step of building an ANN binary classifier using Keras.

## How many nodes should I add to the hidden layers?
The short answer: that is an art. The best number of nodes per hidden layer is found through experimentation.

A starting point if you don't want to be an artist is using __the average between the number of nodes in the input layer and the output layer__. 
  - For example, if we have 11 input neurons and 1 output neuron, then a good starting point for each hidden layer is 6.
  - Then experiment around that number.

## Which activation function to use?
This is also mostly an art. However, certain problems have well defined activation functions.

The [intuition](1-intuition.md#the-activation-function) article has more information on activation functions. 

### Binary Classification
Use 1 output neuron with a `sigmoid` activation function.

### Multi-class Classification 
If you have a 3-class problem:
- One-hot encode the label without dropping any column (the dummy variable trap does not apply in Y).
- Use 3 output neurons with the `softmax` activation function.

## Which loss function to use?
The loss function to choose depends on the activation function of your output layer. There are some pairs of
<output activation functions, loss functions> that play well together.

For example:
- For __binary classification__ -> `sigmoid` output activation -> `binary_crossentropy` loss function
- For __multi-class classification__ -> `softmax` output activation -> `categorical_crossentropy` loss function.

## Selecting the optimizer
- List of commonly used optimizers: Adam, SGD, RMSprop, Adadelta
- The choice of an optimizer typically depends on your problem statement
- See the [intuition notes](1-intuition.md#learning-rate) for some intuition on how to choose the learning rate.
- The authors of the Udemy course suggest going with the `adam` optimizer and the default learning rate unless you
know what you are doing.

### Customizing the learning rate in Keras
If you compile the ann with `optimizer='adam'` it will use a default learning rate.
If you want to customize it you can do the following:
```python
from keras.optimizers import Adam
# .......
# .......
opt = Adam(lr=1e-3)
my_ann.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
```

## Selecting `batch size` and `epochs`
There is no rule of thumb to choose this hyper-parameters. It is an art and also requires experimentation.

## Overfitting control using _dropout regularisation_
__Dropout regularisation__ is a technique for reducing overfitting problems in deep learning.

It works by randomly disabling a fraction of the neurons on each layer at each iteration of the training process.
If we think of each layer of an ANN as some automatically learned representation of the input features, dropping neurons
randomly is equivalent to dropping features randomly. From this angle, this mechanism is similar in spirit as the one used by 
__random forests__ to control overfitting (and ensure generalisation). __Random Forest trees__ are built using 
a randomly chosen sub-set of features. 

### Detecting overfitting
There are 2 ways for detecting overfitting:
1. When we see that the training set accuracy largely outperforms the test set accuracy.
2. When there is a high accuracy variance (or standard deviation) in [cross-validation](3-evaluating-and-tuning-an-ann.md#k-fold-cross-validation).

### Implementing dropout
- When you observer overfitting, it is recommended that you apply dropout to all hidden layers.
- Start with a small fraction of dropout (e.g 0.1) and assess if that fixes the problem. If not, slowly increase the 
fraction and keep assessing.
- Don't go beyond `0.5` or you will risk underfitting.

