"""
Regression Using Decision Tree, Random Tree, Bootstrap Aggregating,
and Boosting.

Copyright (c) 2020 Gabriele Gilardi

References
----------
- Based on project 3 in the Georgia Tech Spring 2020 course "Machine Learning for
  Trading" by Prof. Tucker Balch.
- Course: http://quantsoftware.gatech.edu/CS7646_Spring_2020
- Project: http://quantsoftware.gatech.edu/Spring_2020_Project_3:_Assess_Learners

Characteristics
---------------
- The code has been written and tested in Python 3.6.10.
- Decision and random tree implementation for regression.
- Decision tree: cross-correlation and median used to determine the best
  feature to split and the split value.
- Random tree: best feature and split value determined randomly
- Tree-size reduction can be done by by leaf (defining the lowest number of
  leaves to keep on a branch) and by value (defining the tolerance to group
  close-values leaves).
- The corner case where after a split all data end up in one branch is also
  implemented.
- Bootstrap aggregating (bagging) can be applied to both decision tree and
  random tree.
- AdaBoosting is implemented as boosting algorithm.
- Usage: python test.py <csv-filename>.

Main parameters
---------------
sys.argv[1]
    File name with the dataset passed as argument. Data must be in a csv file,
    with each column a feature and the label in the last column.
0 < split_factor < 1
    Split value between training and test data.
learner_type = dt, rt
    Type of learner (decision tree or random tree).
leaf >= 1
    Lowest number of leaves to keep; any branch with equal or less leaves is
    substituted by a single leaf with a value equal to the averageof the
    removed leaves.
tol >= 0.0
    Tolerance to group leaves based on their labels; any branch where the
    leaves have a value that differ from their average less or equal to this
    tolerance is substituted by a single leaf with a value equal to the average.
bags >= 0
    Number of bags to be used for bootstrap aggregating; no bagging is enforced
    setting this value to zero.
0 < bag_factor <=1
    Number of data in each bag as a fraction of the number of training data.
boost = True, False
    Specify if AdaBoost should be used (True) or not (False).

Examples
--------
All examples are for the file `istanbul.csv`. Correlation results are obtained
averaging 20 runs.

- Reference case: decision tree learner, no reduction, no bagging, no boosting.
- Correlation predicted/actual values: 0.9992 (training), 0.7109 (test).
split_factor = 0.7
learner_type = 'dt'
leaf = 1
tol = 1.0e-6
bags = 0
bag_factor = 1.0
boost = False

- As reference case but using a random tree learner.
- Correlation predicted/actual values: 0.9708 (training), 0.6349 (test).
- Comment: worst results but faster computation.
split_factor = 0.7
learner_type = 'rt'
leaf = 1
tol = 1.0e-6
bags = 0
bag_factor = 1.0
boost = False

- As reference case but changing the number of leaves.
- Correlation predicted/actual values: 0.8917 (training), 0.7843 (test).
- Comment: reduce overfitting and generalize better.
split_factor = 0.7
learner_type = 'dt'
leaf = 10
tol = 1.0e-6
bags = 0
bag_factor = 1.0
boost = False

- As reference case but changing the tolerance.
- Correlation predicted/actual values: 0.9087 (training), 0.7642 (test).
- Comment: reduce overfitting and generalize better.
split_factor = 0.7
learner_type = 'dt'
leaf = 1
tol = 1.0e-2
bags = 0
bag_factor = 1.0
boost = False

- As reference case but using bagging.
- Correlation predicted/actual values: 0.9748 (training), 0.8333 (test).
- Comment: generalize better, using boosting will slightly improve results.
split_factor = 0.7
learner_type = 'dt'
leaf = 1
tol = 1.0e-6
bags = 10
bag_factor = 1.0
boost = False
"""

import sys
import numpy as np
from math import sqrt

import DTLearner as dtl         # Decision tree class
import RTLearner as rtl         # Random tree class
import BagLearner as bl         # Bootstrap aggregating and boosting class

# Parameters
split_factor = 0.7          # Training/test data split
learner_type = 'dt'         # Type of learner
leaf = 1                    # Lowest number of leaves
tol = 1.0e-6                # Tolerance to group close-valued leaves
bags = 0                    # Number of bags
bag_factor = 1.0            # Number of data in each bag
boost = False               # Boosting on/off boolean

# Added for repeatibility
np.random.seed(1)
# Added to avoid message warning "invalid value encountered in true_divide
# c /= stddev[None, :]" during cross-correlation computation
np.seterr(all='ignore')

# Read data from a csv file
if len(sys.argv) != 2:
    print("Usage: python test.py <csv-filename>")
    sys.exit(1)
data = np.loadtxt(sys.argv[1], delimiter=',')

# Build the training and test datasets (randomly)
train_rows = int(split_factor * data.shape[0])
a = np.arange(data.shape[0])
train_idx = np.random.choice(a, size=train_rows, replace=False)
test_idx = np.delete(a, train_idx)
train_data = data[train_idx, :]
test_data = data[test_idx, :]

# Split features (X) and label (Y)
train_X = train_data[:, 0:-1]
train_Y = train_data[:, -1]
test_X = test_data[:, 0:-1]
test_Y = test_data[:, -1]

# Decision Tree
if (learner_type == 'dt'):
    name = 'Decision Tree'

    # No bagging
    if (bags == 0):
        learner = dtl.DTLearner(leaf, tol)

    # With bagging
    else:
        learner = bl.BagLearner(dtl.DTLearner, {"leaf": leaf, "tol": tol},
                                bags, bag_factor, boost)

# Random tree
elif (learner_type == 'rt'):
    name = 'Random Tree'

    # No bagging
    if (bags == 0):
        learner = rtl.RTLearner(leaf, tol)

    # With bagging
    else:
        learner = bl.BagLearner(rtl.RTLearner, {"leaf": leaf, "tol": tol},
                                bags, bag_factor, boost)

else:
    sys.exit("Learner not found")

# Create the model
learner.createModel(train_X, train_Y)

# Predict training and test dataset labels
train_pred = learner.evalData(train_X)
test_pred = learner.evalData(test_X)

# Compute root-mean-square-error and correlation between predicted and actual
# labels for training and test dataset
train_rmse = sqrt(((train_Y - train_pred) ** 2).sum() / len(train_pred))
train_corr = np.corrcoef(train_Y, train_pred)
test_rmse = sqrt(((test_Y - test_pred) ** 2).sum() / len(test_pred))
test_corr = np.corrcoef(test_Y, test_pred)

# Print info and results
print('\nType: ', name)
print('Dataset:', sys.argv[1])
print('Number of features:', train_X.shape[1])
print('Training data:', train_X.shape[0])
print('Test data:', test_X.shape[0])
if (bags > 0):
    print('Number of bags: ', bags)
    print('Data in each bag:', int(bag_factor * train_X.shape[0]))
print('\n       Train.  Test')
print("RMSE:  {0:6.4f}   {1:6.4f}".format(train_rmse, test_rmse))
print("Corr:  {0:6.4f}   {1:6.4f}".format(train_corr[0, 1], test_corr[0, 1]))
