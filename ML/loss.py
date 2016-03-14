#!/usr/bin/python3

"""loss.py: Contains various loss functions."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np


def svm_loss(scores, y):
    """Calculates the hinge loss."""

    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss = np.sum(margins)

    return loss


def softmax_loss(scores, y):
    """Calculates the cross-entropy loss."""

    probability = np.exp(scores[y]) / np.sum(np.exp(scores))
    loss = -np.log(probability)

    return loss


def get_loss(X, Y, W, regularization=1):
    loss = 0

    for x, y in zip(X, Y):
        scores = W @ x
        loss += svm_loss(scores, y)

    loss /= X.size

    # Regularizes the loss function using L2 regularization.
    loss += regularization * (W**2).sum()

    return loss