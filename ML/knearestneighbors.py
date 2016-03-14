#!/usr/bin/python3

"""knearestneighbors.py: Implements the k-nearest neighbors classifier."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np
from mathhelpers import get_p_norm


class KNearestNeighbors:
    def train(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, X, k):
        test_size = X.shape[0]
        Y = np.empty(test_size)

        for index in range(test_size):
            # Measures the distances between X and every element of the training set.
            distances = get_p_norm(self.X - X[index, :], 2)

            # Finds the classes of the k-nearest neighbors.
            k_nearest_Y = self.Y[np.argsort(distances)[:k]]

            # Selects the most common class.
            Y[index] = np.bincount(k_nearest_Y).argmax()

        return Y