import numpy as np


def svm_loss(x, y, W):
	"""Calculates the hinge loss."""

	scores = W @ x
	margins = np.maximum(0, scores - scores[y] + 1)
	margins[y] = 0
	loss = np.sum(margins)

	return loss


def softmax_loss(x, y, W):
	"""Calculates the cross-entropy loss."""

	scores = W @ x
	probability = np.exp(scores[y]) / np.sum(np.exp(scores))
	loss = -np.log(probability)

	return loss


def get_loss(X, Y, W, regularization=1):
	loss = 0

	for x, y in zip(X, Y):
		loss += svm_loss(x, y, W)

	loss /= X.size

	# Regularizes the loss function using L2 regularization.
	loss += regularization * (W**2).sum()

	return loss