# Recurrent Neural Network (RNN) Intuition

## What are Recurrent Neural Networks Used for?

RNNs are used in problems where knowing the previous context is important to determine the next output.
For example, if a text predictor wants to predict the next word on the phrase _"Regarding food, I love \_"_,
the previous words _"Regarding food, I love"_ are critical context for predicting what comes next. 

The technical term for this type of problems is __Time Series Analysis__.

Natural Language Processing is a field that heavily uses RNNs since they naturally model some key characteristics of 
natural language that other types of networks struggle with:
- In natural language, the word order is critical in giving the text meaning. There is no easy way to model order 
dependencies in ANNs or CNNs.
- Natural Language has "long distance effects". These are inherently a time series problem. 
   - e.g. _"I, the president of the United States, AM going to declare"_. In this example,  _"AM"_ is dependent on the
   word _"I"_.
- Inputs in Natural Language problems of varying length because texts are of varying length. ANNs and
CNNs take fixed size inputs.

## Structure of a RNN

RNNs are typically drawn in 2 forms: _rolled_ and _un-rolled_ versions. The figure below explains the structure
using the _un-rolled_ representation.
 
![Simplified RNN Structure][simplified-rnn-structure]

The key characteristic of RNNs is that neurons are connected to themselves through time. This means that a 
neuron at time `t+1` uses 2 sources of inputs:
1. The input from the `input sequence`.
1. The output from time `t`.

Don't be fooled by the simplified flat representation above. RNNs are multidimensional. A more precise depicting
of how they are structured would be:
![Multidimensional RNN Structure][multidimensional-rnn-structure]
 
 
## Example Applications of RNNs

RNN architecture varies depending on the problem. Some RNNs take a single input, some others take a sequence of inputs.
A similar thing happens with the outputs.

The diagram below shows multiple variants of RNN architectures being applied to solve different types of problems.

![Sample RNN applications][sample-rnn-applications]

## The Vanishing Gradient Problem
### A Solution to the Vanishing Gradient Problem: Long Short-Term Memory (LSTM)
- Weights as we have seen them in ANNs and CNNs can be seen as a Long Term Memory

[simplified-rnn-structure]: ./simplified-rnn-structure.png
[multidimensional-rnn-structure]: ./multidimensional-rnn-structure.png
[sample-rnn-applications]: ./sample-rnn-applications.png