"""
Regression and Classification Using Decision Tree, Random Tree, Bootstrap
Aggregating, and Boosting.

Copyright (c) 2020 Gabriele Gilardi
"""

import numpy as np

class RTLearner:

    def __init__(self, leaf=1, tol=1.0e-6):
        """
        leaf        Lowest number of leaves
        tol         Tolerance to group close-valued leaves
        """
        self.leaf = leaf
        self.tol = tol

    def buildTree(self, X, Y):
        """
        Builds the decision-tree table.

        column 0 = feature used for the split (-1 indicates a leaf)
        column 1 = split value
        column 2 = relative position left branch (0 indicates no left branch)
        column 3 = relative position right branch (0 indicates no right branch)
        """
        # Return the mean if equal or less than the lowest number of leaves
        if (X.shape[0] <= self.leaf):
            return np.array([-1, Y.mean(), 0, 0])

        # Return the mean if all remaining leaves have close values
        Ym = Y.mean()
        deltaY = np.absolute(Y-Ym)
        if (all(deltaY <= self.tol)):
            return np.array([-1, Ym, 0, 0])

        # Keep splitting
        else:
            # Randomly pick a feature and split
            idx = np.random.randint(0, X.shape[1])
            i = np.random.randint(0, X.shape[0], size=2)
            split_value = (X[i[0], idx] + X[i[1], idx]) / 2.0

            # Build the left branch dataset
            X_left = X[X[:, idx] <= split_value]
            Y_left = Y[X[:, idx] <= split_value]

            # Return the mean if there is no split (because all data end up in
            # the left branch).
            if (X_left.shape[0] == X.shape[0]):
                return np.array([-1, Y_left.mean(), 0, 0])

            # Keep splitting
            else:
                # Build the right branch dataset
                X_right = X[X[:, idx] > split_value]
                Y_right = Y[X[:, idx] > split_value]

                # Search the two new branches
                left_branch = self.buildTree(X_left, Y_left)
                right_branch = self.buildTree(X_right, Y_right)

                # Return the sub-tree table
                k = divmod(len(left_branch), 4)[0]
                root = np.array([idx, split_value, 1, k+1])
                return np.concatenate((root, left_branch, right_branch))

    def createModel(self, X, Y):
        """
        Wrapper for building the decision-tree table.
        """
        # Build the tree-table as 1-dim array
        a = self.buildTree(X, Y)

        # Reshape it as an (n_row, 4) matrix
        n_row = divmod(len(a), 4)[0]
        self.treeTable = a.reshape(n_row, 4)

    def evalData(self, X):
        """
        Evaluates a dataset of features with the created decision-tree table.
        
        column 0 = feature used for the split (-1 indicates a leaf)
        column 1 = split value
        column 2 = relative position left branch (0 indicates no left branch)
        column 3 = relative position right branch (0 indicates no right branch)
        """
        n = X.shape[0]              # Number of data to evaluate
        pred_Y = np.empty(n)        # Allocate prediction array

        # Loop over the dataset
        for i in range(n):

            # Start from the root node of the decision-tree table
            row = 0
            feature = int(round(self.treeTable[0, 0]))

            # Move along the decision-tree table until a leaf is found
            while (feature != -1):

                # Next node is on the left branch
                if (X[i, feature] <= self.treeTable[row, 1]):
                    delta = int(round(self.treeTable[row, 2]))

                # Next node is on the right branch
                else:
                    delta = int(round(self.treeTable[row, 3]))

                # Get the feature of the next node
                row += delta
                feature = int(round(self.treeTable[row, 0]))

            # Set the leaf value as predicted value
            pred_Y[i] = self.treeTable[row, 1]

        return pred_Y
