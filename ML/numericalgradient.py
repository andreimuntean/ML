#!/usr/bin/python3

"""mathhelpers.py: Provides math functions."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np


def get_numerical_grad(f, x, h=0.001):
	"""Computes the numerical gradient of the specified function."""

	initial_fx = f(x)
	grad = np.zeros(x.shape)

	for index in range(0, x.size):
		x[index] += h
		grad[index] = (f(x) - initial_fx) / h
		x[index] -= h

	return grad