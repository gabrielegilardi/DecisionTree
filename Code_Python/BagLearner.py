"""
Regression and Classification Using Decision Tree, Random Tree, Bootstrap
Aggregating, and Boosting.

Copyright (c) 2020 Gabriele Gilardi
"""

import numpy as np

class BagLearner():

    def __init__(self, learner, kwargs={}, bags=10, bag_factor=1, boost=False):
        """
        learner         Learner type
        kwargs          Parameters passed to the learner
        bags            Number of bags to be used for bootstrap aggregating
        bag_factor      Number of data in each bag as a fraction of the number
                        of training data
        boost           Specify if boosting should be used or not
        """
        self.kwargs = kwargs
        self.bags = bags
        self.bag_factor = bag_factor
        self.boost = boost

        # Initialize the ensemble of learners
        self.ensemble = []
        for i in range(self.bags):
            self.ensemble.append(learner(**self.kwargs))

    def createModel(self, X, Y):
        """
        Builds the ensemble of learners.
        """
        n = X.shape[0]                  # Number of training data
        m = int(self.bag_factor*n)      # Number of data in each bag
        a = np.arange(n)                # Index vector of the training data
        prob = np.ones(n) / n           # Initial probability distribution
        pred_Y = np.zeros(n)            # Initialize running sum of the
                                        # predictions

        # Build the model for each learner
        for i in range(self.bags):

            # Randomly pick (with replacement) 'm' rows from the training
            # data using the probability distribution 'prob' (always uniform
            # distribution if no boosting) and train the learner with the
            # corresponding bag of data
            idx = np.random.choice(a, size=m, replace=True, p=prob)
            self.ensemble[i].createModel(X[idx, :], Y[idx])

            # If boosting is requested the probabilities are proportional
            # to the absolute value between actual label and the current
            # prediction's average
            if (self.boost):

                # Add the prediction of the current learner to the previous
                # predictions
                pred_Y += self.ensemble[i].evalData(X)

                # Compute the prediction error in absolute value
                error = np.absolute(Y - pred_Y / (i + 1))

                # Normalize so probabilities will add to 1
                prob = error / error.sum()

    def evalData(self, X):
        """
        Evaluates a dataset of features for the ensemble of learners.
        """
        pred_Y = np.zeros(X.shape[0])    # Initialize prediction array

        # Add the prediction of each learner in the ensemble
        for i in range(self.bags):
            pred_Y += self.ensemble[i].evalData(X)

        # Return the average
        return (pred_Y / self.bags)
