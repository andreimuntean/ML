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
    accuracy = np.sum(Y == predictions) / X.shape[0]

    return accuracy


# Loads the training and test data.
training_data, test_data = load_cifar10(path.join('data', 'cifar10'))

# Formats the training data.
training_X = np.concatenate([training_data[0]['data'],
                             training_data[1]['data'],
                             training_data[2]['data'],
                             training_data[3]['data'],
                             training_data[4]['data']])
training_Y = np.concatenate([training_data[0]['labels'],
                             training_data[1]['labels'],
                             training_data[2]['labels'],
                             training_data[3]['labels'],
                             training_data[4]['labels']])

# Formats the test data.
test_X = test_data['data']
test_Y = test_data['labels']

# Trains the classifier.
model = KNearestNeighbors()
model.train(training_X, training_Y)

# Tests the classifier.
k = 3
test_count = 50
test_accuracy = get_accuracy(model,
                             test_X[:test_count],
                             test_Y[:test_count],
                             k)
training_accuracy = get_accuracy(model,
                                 training_X[:test_count],
                                 training_Y[:test_count],
                                 k)

print('For k = {}:'.format(k))
print('Test data accuracy: {}%.'.format(test_accuracy * 100))
print('Training data accuracy: {}%.'.format(training_accuracy * 100))