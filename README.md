
# Regression Using Decision Tree, Random Tree, Bootstrap Aggregating, and Boosting

## Reference

Based on [project 3](http://quantsoftware.gatech.edu/Spring_2020_Project_3:_Assess_Learners) in the Georgia Tech Spring 2020 course [Machine Learning for Trading](http://quantsoftware.gatech.edu/CS7646_Spring_2020) by Prof. Tucker Balch.

## Characteristics

- The code has been written and tested in Python 3.6.10.
- Decision and random tree implementation for regression.
- Decision tree: cross-correlation and median used to determine the best feature to split and the split value.
- Random tree: best feature and split value determined randomly.
- Tree-size reduction can be done by by leaf (defining the lowest number of leaves to keep on a branch) and by value (defining the tolerance to group close-values leaves).
- The corner case where after a split all data end up in one branch is also implemented.
- Bootstrap aggregating (bagging) can be applied to both decision tree and random tree.
- AdaBoosting is implemented as boosting algorithm.
- Usage: *python test.py csv-filename*.

## Parameters

`sys.argv[1]` File name with the dataset passed as argument. Data must be in a csv file, with each column a feature and the label in the last column.

`split_factor` Split value between training and test data.

`learner_type` Type of learner (decision tree or random tree).

`leaf` Lowest number of leaves to keep; any branch with equal or less leaves is substituted by a single leaf with a value equal to the average of the removed leaves.

`tol` Tolerance to group leaves based on their labels; any branch where the leaves have a value that differ from their average less or equal to this tolerance is substituted by a single leaf with a value equal to the average.

`bags` Number of bags to be used for bootstrap aggregating; no bagging is enforced setting this value to zero.

`bag_factor` Number of data in each bag as a fraction of the number of training data.

`boost` Specify if boosting should be used or not.

### Examples

All examples are for the file `istanbul.csv`. Correlation results are obtained averaging 20 runs.

- Reference case: decision tree learner, no tree reduction, no bagging, no boosting. Correlation predicted/actual values: 0.9992 (training), 0.7109 (test).

```python
split_factor = 0.7
learner_type = 'dt'
leaf = 1
tol = 1.0e-6
bags = 0
bag_factor = 1.0
boost = False
```

- As reference case but using a random tree learner. Correlation predicted/actual values: 0.9708 (training), 0.6349 (test). Comment: worst results but faster computation.

```python
split_factor = 0.7
learner_type = 'rt'
leaf = 1
tol = 1.0e-6
bags = 0
bag_factor = 1.0
boost = False
```

- As reference case but changing the number of leaves. Correlation predicted/actual values: 0.8917 (training), 0.7843 (test). Comment: reduce overfitting and generalize better.

```python
split_factor = 0.7
learner_type = 'dt'
leaf = 10
tol = 1.0e-6
bags = 0
bag_factor = 1.0
boost = False
```

- As reference case but changing the tolerance. Correlation predicted/actual values: 0.9087 (training), 0.7642 (test). Comment: reduce overfitting and generalize better.

```python
split_factor = 0.7
learner_type = 'dt'
leaf = 1
tol = 1.0e-2
bags = 0
bag_factor = 1.0
boost = False
```

- As reference case but using bagging. Correlation predicted/actual values: 0.9748 (training), 0.8333 (test). Comment: generalize better, using boosting will slightly improve results.

```python
split_factor = 0.7
learner_type = 'dt'
leaf = 1
tol = 1.0e-6
bags = 10
bag_factor = 1.0
boost = False
```
