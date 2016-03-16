#!/usr/bin/python3

"""loss.py: Contains various loss functions."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np


def get_svm_loss(X, Y, W, margin=1, regularization=1):
    """Calculates the hinge loss."""

    # Computes the scores of every x and w.
    scores = X @ W.T

    # Computes the margin loss.
    Y_rows = np.arange(Y.size)
    margin_loss = np.maximum(0, (scores.T - scores[Y_rows, Y]).T + margin)
    margin_loss[Y_rows, Y] = 0

    # Computes the loss and applies L2 regularization.
    loss = np.sum(margin_loss) / X.shape[0] + regularization * np.sum(W**2)

    # Computes the gradients for this function.
    grads = np.zeros(W.shape)

    for index in range(X.shape[0]):
        # For brevity.
        x = X[index, :]
        y = Y[index]
        margin_loss_i = margin_loss[index, :]

        # Computes the gradient.
        grad = np.zeros(W.shape) + x
        grad = (grad.T * margin_loss_i).T
        grad[y, :] = -np.count_nonzero(margin_loss_i) * x
        
        # Accumulates the gradient.
        grads += grad

    # Finds the overall gradient by calculating the mean.
    grads /= X.shape[0]

    return loss, grads


def get_softmax_loss(X, Y, W, regularization=1):
    """Calculates the cross-entropy loss."""

    # Computes the scores of every x and w.
    scores = X @ W.T

    # Shrinks the scores to prevent exponentiation overflows.
    scores /= np.max(scores, axis=1, keepdims=True)

    # Applies the softmax function to compute the probabilities.
    Y_rows = np.arange(Y.size)
    probabilities = np.exp(scores[Y_rows, Y]) / np.sum(np.exp(scores), axis=1)

    # Computes the loss and applies L2 regularization.
    loss = np.sum(-np.log(probabilities)) / X.shape[0] + regularization * np.sum(W**2)

    return loss