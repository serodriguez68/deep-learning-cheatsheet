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
- Inputs in many NLP problems need to be of varying length because texts are of varying length. ANNs and
CNNs only take fixed size inputs.

## Structure of a RNN

RNNs are typically drawn in 2 forms: _rolled_ and _un-rolled_ versions. The figure below explains the structure
using the _un-rolled_ representation.
 
![Simplified RNN Structure][simplified-rnn-structure]

The key characteristic of RNNs is that neurons are connected to themselves through time. This means that a 
neuron at time `t+1` uses 2 sources of inputs:
1. The input from the `input sequence`.
1. The output from time `t`.

Don't be fooled by the simplified flat representation above. RNNs are multidimensional. A more precise representation
of how they are structured would be:
![Multidimensional RNN Structure][multidimensional-rnn-structure]
 
 
## Example Applications of RNNs

RNN architecture varies depending on the problem. Some RNNs take a single input, some others take a sequence of inputs.
A similar thing happens with the outputs.

The diagram below shows multiple variants of RNN architectures being applied to solve different types of problems.

![Sample RNN applications][sample-rnn-applications]

## The Difficulties of Training Deep Networks: The Vanishing and Exploding Gradient Problems

Neural networks get increasingly harder to train as we add more layers. The `gradient`, the main piece of information
for updating the weights, gets __unstable__ as the number of layers increase due to compounded derivative calculations.
- Remember that the `gradients` in `layer l` are calculated as the product of the derivatives for all layers from `l+1`
to the last layer.  For example, in a 100-deep network, `gradient_l1 ~= some_derivative_l2 * some_derivative_l3 * ... some_derivative_l100`.
- __Vanishing Gradient:__ When many `some_derivative_lX` are `< 1` their product will be very close to `0`. This results in `gradient_l1` 
being close to `0`, so the `weights` in `l1` barely get updated during back propagation and get stuck. This means that early layers in the
network are hard to train and require many epochs and lots of training data.
- __Exploding Gradient__: When many `some_derivative_lX` are `> 1` their product will be very big. This results
in `gradient_l1` being very big and causing a violent and unstable update of the `weights` in `l1`. This big
"jumps" in weights make the training unstable and the network might never find the optimal due to these swings.

If this doesn't make any sense, I suggest [DeepLizard's youtube video][deep-lizard-vanishing-gradient-problem] on this topic. 

This problem is not exclusive to RNNs. However, RNNs are very prone to it as their recursive nature makes them
"arbitrarily deep".

![The vanishing gradient problem in an RNN][vanishing-gradient-problem-rnn]


### Solutions to these problems

In general, both problems are governed by the same mechanism and can be partially solved by doing smart weight 
initialization that ensures that the gradients don't get too small or too big.

A very popular way of doing this "smart" initialization is called __Xavier Initialization__ (aka Glorot Initialization).
__Xavier Initialization__ is a _per layer_ initialization technique that initializes the weights randomly with a normal
distribution around 0 and standard deviation that changes per layer. To know more details about this
initialization see [DeepLizard's video on it][deep-lizard-xavier-initialization].

`Keras` supports Xavier (Glorot) initialization as shown next. Moreover, if we don't specify the type of
`kernel_initializer`, it will use `glorot_uniform` by default.

```python
Dense(32, activation=relu, kernel_initializer='glorot_uniform')
``` 

There are other solutions to these problems that are specific for the type of problem we face: explosion or vanishing.
The details of most of these solutions are out of the scope of these notes.

### Some specific solutions for the Exploding Gradient Problem
- __Truncated Back Propagation__: stop back propagation after some amount of layers.
- __Penalties__
- __Gradient Clipping__: Set a fixed max value for the gradient and back propagate that value when the calculated gradient exceeds it.
 
### Some specific solutions for the Vanishing Gradient Problem
- __Echo State Networks__: A network architecture that helps with the problem (out of scope of this summary). 
- __Long Short-Term Memory (LSTM) Networks__: Another network architecture that tackles this problem. This architecture
is very popular for RNNs and we will discuss it in detail in the [next section][lstm-section].
  

## Long Short-Term Memory (LSTM) Networks

This summary is heavily based on the following article. 
I strongly recommend reading it: [Understanding LSTM Networks by Christopher Olah][understanding-lstm-colah]

LSTM networks are a particular architecture of the more general RNN category. They are of particular importance because 
almost all exciting results from RNNs have been achieved using this architecture.

### Motivation

An RNNs main feature taking into account previous inputs and predictions to perform the current time-step prediction.
 
When the previous relevant context happened recently, "Vanilla" RNNs have no problem using that to inform the current
prediction.  For example, in the if we were predicting the next word in the phrase _"the __clouds__ are in the `sky`"_,
__"clouds"__ happens very close to the place where `sky` is being predicted.

However, in many applications (including text) context has a long distance effect. In these circumstances, 
"Vanilla RNNs" struggle to "remember" what happened many time steps ago. 

Continuing with the text prediction example,
if the phrase was _"I grew up in __France__, the birthplace of civil liberties. I speak fluent `French`"_. __"France__"
is critical context to predict `French`, but it happened so long ago that a "Vanilla RNN" won't be able to "remember"
it to narrow down the prediction.  This is why it is said that Vanilla RNNs have short term memory.

- We've been talking about the short term memory problems of "Vanilla RNNs" from the inference point of view. However,
they have the same problem at training time.  During training, "Vanilla RNNs" are unable to connect related concepts if
they have a long distance between them.

### LSTM Intuition

LSTMs are specifically designed to tackle the short-term memory problem.  In fact, having long-term memory is their
default behaviour.

An LSTM is a group of interconnected learned single-layer neural networks and point-wise operations that are grouped
together in a "repeating group".

Each of these neural networks and point-wise operations serve a particular purpose in the architecture and are explained
below. For now, lets get familiar with the notation:

![LSTM notation][lstm-notation]
 
#### The Memory Pipeline - The core idea 
The core idea behind LSTMs is the __"memory cell / pipeline"__. Through the __memory pipeline__, information flows easily
and mostly unchanged across the multiple steps of the RNN.  Changes to the memory only happen when the gates
determine so (see below).

> Side note: Intuitively, since the information in the pipeline doesn't change much, the derivative over time stays close to 1. This
is why LSTMs are less prone to suffer from vanishing or exploding gradient problems.
    
![The memory Pipeline - The core idea behind LSTMs][lstm-core-idea]


#### Step-by-step walk through
![LSTM walk through part 1][lstm-walk-through-pt1]
![LSTM walk through part 2][lstm-walk-through-pt2]  


### LSTM Variations

There are many variants of the LSTM architecture shown above. Most of them are slight variations of
this architecture and have similar performance.

- Some popular variants are: Peephole connections, unified forget and remember decisions, Gated Recurring Units (GRUs), among many others.
- Go to [Christopher Olah's article to read about these popular variants][understanding-lstm-colah].
- [Greff et al][greff-et-al] published a paper showing that most variations are about the same.
- [Jozefowicz et al][jozefowicz-et-al] published a paper comparing many architectures and shows that some work better
on certain tasks.

## Beyond LSTMs - Attention Models

LSTMs were a step forward in RNN performance. The next big step in RNN performance are __attention RNNs__.
[Chirstopher Colah has an article on Attention RNNs][attention-colah] and other forms of RNN augmentation.

## Other Additional Reading
- If you want to understand even deeper how each of the operations inside the LSTM repreating unit works,
I recommend reading ["Understanding LSTM and its diagrams"][understanding-lstm-and-its-diagrams]. 
- [Visualizing and Understanding Recurrent Networks by Andrej Karpathy (2015)][karpathy].


[simplified-rnn-structure]: ./simplified-rnn-structure.png
[multidimensional-rnn-structure]: ./multidimensional-rnn-structure.png
[sample-rnn-applications]: ./sample-rnn-applications.png
[deep-lizard-vanishing-gradient-problem]: https://youtu.be/qO_NLVjD6zE
[vanishing-gradient-problem-rnn]: ./vanishing-gradient-problem-rnn.png
[deep-lizard-xavier-initialization]: https://www.youtube.com/watch?v=8krd5qKVw-Q
[lstm-section]: #long-short-term-memory-lstm-networks
[understanding-lstm-colah]: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
[lstm-notation]: ./lstm-notation.png
[lstm-core-idea]: ./lstm-core-idea.png
[lstm-walk-through-pt1]: ./lstm-walk-through-pt1.png
[lstm-walk-through-pt2]: ./lstm-walk-through-pt2.png
[greff-et-al]: https://arxiv.org/pdf/1503.04069.pdf
[jozefowicz-et-al]: http://proceedings.mlr.press/v37/jozefowicz15.pdf
[attention-colah]: https://distill.pub/2016/augmented-rnns/
[understanding-lstm-and-its-diagrams]: https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714
[karpathy]: https://arxiv.org/pdf/1506.02078.pdf
