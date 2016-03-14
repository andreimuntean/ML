#!/usr/bin/python3

"""datahelpers.py: Provides functions for handling data."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np
import pickle
from os.path import join


def read_data(path):
    """Reads the specified pickled file."""

    file = open(path, 'rb')
    data = pickle.load(file, encoding='latin1')
    file.close()

    return data


def load_cifar10(path):
    """Reads the CIFAR-10 data set from the specified path."""

    test_data = read_data(join(path, 'test_batch'))
    training_data = []

    for index in range(5):
        batch_path = join(path, 'data_batch_{}'.format(index + 1))
        batch = read_data(batch_path)
        training_data.append(batch)

    return training_data, test_data