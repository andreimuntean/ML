#!/usr/bin/python3

"""gradientdescent_test.py: Tests an implementation of mini-batch gradient descent."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np
from datahelpers import load_cifar10
from loss import get_svm_loss, get_softmax_loss
from mathhelpers import get_numerical_grad
from os import path


# Loads the training and test data.
X, Y, test_X, test_Y = load_cifar10(path.join('data', 'cifar10'))

# Appends the bias term.
X = np.column_stack((np.ones(X.shape[0]), X))
test_X = np.column_stack((np.ones(test_X.shape[0]), test_X))

# Configures the hyperparameters.
batch_size = 128
step_size = 0.00000001

# Generates the initial weights.
W = np.random.randn(10, 3073) * 0.0001

for iteration in range(1, 51):
	for start, end in zip(range(0, X.shape[0], batch_size), range(batch_size, X.shape[0], batch_size)):		
		loss, grads = get_svm_loss(X[start : end, :], Y[start : end], W)		
		W -= step_size * grads

	print('Iteration: {} -- Loss: {}'.format(iteration, loss))

predictions = np.argmax(X @ W.T, axis=1)
training_accuracy = np.mean(Y == predictions)

predictions = np.argmax(test_X @ W.T, axis=1)
test_accuracy = np.mean(test_Y == predictions)

print('Test data accuracy: {}%.'.format(test_accuracy * 100))
print('Training data accuracy: {}%.'.format(training_accuracy * 100))