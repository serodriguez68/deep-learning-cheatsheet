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
[This annotated code](./../../annotated-code/Volume%201%20-%20Supervised%20Deep%20Learning/Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/ann_churn_classifier.py)
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