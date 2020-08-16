# Overview

This cheat sheet covers all of the __coding, intuition and application__ aspects of the foundational deep learning concepts. 
This works assumes that you know the basics of neural networks, and it is intended to be a quick
reference on their intuition and on how to use them using Python libraries like Keras.

This work does not explain the mathematical grounding behind deep learning, but it does give some intuition.

The work is based on a mixture of different resources. Notably: 
- Kirill Eremenko's and Hadelin de Ponteve's 
[Deep Learning A-Z™: Hands-On Artificial Neural Networks](https://www.udemy.com/course/deeplearning/) course
on Udemy. 
- The University of Melbourne's Postgraduate course on [Statistical Machine Learning](https://handbook.unimelb.edu.au/2020/subjects/comp90051).

# Table of Content

This is a work in progress and finishing all topics I want to cover will take a while. However, 
this TOC points to the sections that I have finalized.

### __Volume 1: Supervised Deep Learning__

- Part 1 - Artificial Neural Networks
  - [Intuition](./notes/Volume%201%20-%20Supervised%20Deep%20Learning/Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/1-intuition.md)
  - [Coding an ANN](./notes/Volume%201%20-%20Supervised%20Deep%20Learning/Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/2-coding-an-ann.md)
  - [Evaluation and Tuning](./notes/Volume%201%20-%20Supervised%20Deep%20Learning/Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/3-evaluating-and-tuning-an-ann.md)
  - [Code Example: Customer Churn Classifier using an ANN](./annotated_code/volume_1_supervised_deep_learning/part_1_artificial_neural_networks/ann_churn_classifier.py)
- Part 2 - Convolutional Neural Networks
  - [Intuition](./notes/Volume%201%20-%20Supervised%20Deep%20Learning/Part%202%20-%20Convolutional%20Neural%20Networks%20(CNN)/1-intuition.md)
  - [Coding a CNN](./notes/Volume%201%20-%20Supervised%20Deep%20Learning/Part%202%20-%20Convolutional%20Neural%20Networks%20(CNN)/2-coding-a-cnn.md)
  - [Code Example: Binary Image Classification with Data Augmentation using a CNN](./annotated_code/volume_1_supervised_deep_learning/part_2_convolutional_neural_network/cnn_image_classifier.py)
- Part 3 - Recurrent Neural Networks and LSTMs
  - [Intuition](./notes/Volume%201%20-%20Supervised%20Deep%20Learning/Part%203%20-%20Recurrent%20Neural%20Networks%20(RNN)/1-intuition.md)
  - [Coding an LSTM](./notes/Volume%201%20-%20Supervised%20Deep%20Learning/Part%203%20-%20Recurrent%20Neural%20Networks%20(RNN)/2-coding-an-lstm.md)
  - [Code Example: Time Series Prediction of Google's Stock Price using an LSTM](./annotated_code/volume_1_supervised_deep_learning/part_3_recurrent_neural_networks/lstm_stock_trend_regression.py)

### __Volume 2: Unsupervised Deep Learning__
> All of this section is yet to be done
- Part 4 - Self Organising Maps (SOM)
- Part 5 - Boltzmann Machines (BM)
- Part 6 - Dimensionality Reduction with Autoencoders

### __Miscellaneous__: stuff that applies to deep learning in general
 - [Optimizing CPU-Based Deep Learning on Intel Processors](notes/miscellaneous/1-optimizing-cpu-deep-learning-on-intel-processors.md)