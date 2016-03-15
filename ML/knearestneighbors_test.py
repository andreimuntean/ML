#!/usr/bin/python3

"""knearestneighbors.py: Tests the k-nearest neighbors classifier."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np
from datahelpers import load_cifar10
from knearestneighbors import KNearestNeighbors
from os import path


def get_accuracy(model, X, Y, k=3):
    predictions = model.predict(X, k)
    accuracy = np.mean(Y == predictions)

    return accuracy


# Loads the training and test data.
X, Y, test_X, test_Y = load_cifar10(path.join('data', 'cifar10'))

# Trains the classifier.
model = KNearestNeighbors()
model.train(X, Y)

# Tests the classifier.
k = 3
test_count = 50
test_accuracy = get_accuracy(model,
                             test_X[:test_count],
                             test_Y[:test_count],
                             k)
training_accuracy = get_accuracy(model,
                                 X[:test_count],
                                 Y[:test_count],
                                 k)

print('For k = {}:'.format(k))
print('Test data accuracy: {}%.'.format(test_accuracy * 100))
print('Training data accuracy: {}%.'.format(training_accuracy * 100))