# Part 1 - Data Preprocessing
# Part 1 HAS EXACTLY THE SAME CODE AS PART 1 IN ann_churn_classifier.py
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

# Part 2 - Evaluating, Improving and Tuning the ANN
# ----------------------------
# Evaluating the ANN using k-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from annotated_code.volume_1_supervised_deep_learning.part_1_artificial_neural_networks.evaluating_and_tuning.build_classifier_ann import build_classifier_ann
# To be able to parallelize the cross_val_score evaluation, sklearn needs the classifier builder to be independent of
# the file that will run the parallell jobs. That is why we had to extract build_classifier_ann to an external file

classifier_ann = KerasClassifier(build_fn=build_classifier_ann, batch_size=10, epochs=100)
# KerasClassifier wraps a keras network exposing an sk-learn API.
#   build_fn: Accepts a function that returns the ann classifier we want to evaluate.
#   The other args are training args.

accuracies = cross_val_score(estimator=classifier_ann, X=X_train, y=y_train, cv=10, n_jobs=-1)
#   estimator: The classifier that is going to be tested using cross validation.
#   X: data on top of which data cross-validation will be done
#   y: labels for X
#   cv: number of folds. k in k-fold cross validation (10 is very typical)
#   n_jobs: number of CPUs that are going to be used for parallel cross validation. -1 means all CPUs.

# Using X_train instead of all X to strictly keep X_test as an unseen sample that can be used to assess accuracy on
# unseen data (cross validation trains model on the whole data at the end).

mean_accuracy = accuracies.mean()
accuracy_std = accuracies.std()
print('Mean accuracy: ' + str(mean_accuracy))
print('Standard deviation of accruacy: ' + str(accuracy_std))
