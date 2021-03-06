#!/usr/bin/python3

"""mathhelpers.py: Provides math functions."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np


def get_entropy(x):
    """Computes the entropy of the specified array."""

    p_x = np.unique(x, return_counts=True)[1] / x.size
    entropy = -np.sum(p_x * np.log2(p_x))

    return entropy


def get_numerical_grad(f, x, h=0.00001):
	"""Computes the numerical gradient of the specified function."""

	initial_fx = f(x)
	grad = np.zeros(x.shape)

	for index, value in np.ndenumerate(x):
		# Evaluates f(x + h).
		previous_x = x[index]
		x[index] += h
		grad[index] = (f(x) - initial_fx) / h
		x[index] = previous_x

	return grad


def get_p_norm(x, p):
    """Computes the p-norm of the specified tensor."""

    return np.sum((np.abs(x)**p), axis=1)**(1/p)