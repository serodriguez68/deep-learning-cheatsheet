# Evaluating and Tuning an ANN

## Motivation
Using a simple single train-test split evaluation as shown in the [`ann_churn_classifier code`](../../../annotated_code/volume_1_supervised_deep_learning/part_1_artificial_neural_networks/ann_churn_classifier.py)
is a good way to "quickly" iterate over the different features and network arrangements until we
find a candidate architecture we feel confident about.

However, using a single train-test split as a broader measure of performance has a problem. The training process of an 
ANN is non deterministic and dependent on the training data from the split.  Hence, the accuracy results tend to have a high variance. 

As a result, once we want to seriously evaluate the performance of a candidate architecture, we might
need to resort to more robust evaluation strategies like __k-fold cross validation__.

## K-fold cross validation
![cross validation](cross_validation.png)

- The data is first split into training and test set
- The __k-fold cross validation__ is run on the training set as shown on the image above.
- If k = 5 the model is trained 5 + 1 times
   - 5 times each leaving a different fold out. Accuracy is calculated for each training.
   - A 6th time to train the final model with __all the training data__.
- The list of 5 accuracies allows us to calculate the accuracy's __average__ and the __standard deviation / variance__.  This is a much more robust evaluation than a single train-test split.
- The __test set__ is still reserved as a strictly unseen piece of data that can be used to do
a final evaluation of the final model (e.g. the 6th iteration trained with all the data).

## Hyperparameter tuning
This is mostly done through grid search.  In `sklearn` GridSearchCV internally uses k-fold cross validation.
See [the code](#Code) for full details on how to do this.

## Code
See these concepts in action [here](../../../annotated_code/volume_1_supervised_deep_learning/part_1_artificial_neural_networks/evaluating_and_tuning/evaluating_and_tuning_ann.py)