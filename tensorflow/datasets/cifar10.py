# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os

import numpy as np

from .utils import random_balanced_partitions, random_partitions


class Cifar10ZCA:
    DATA_PATH = os.path.join('data', 'images', 'cifar', 'cifar10', 'cifar10_gcn_zca_v2.npz')
    VALIDATION_SET_SIZE = 5000  # 10% of the training set
    UNLABELED = -1

    def __init__(self, data_seed=0, n_labeled='all', test_phase=False):
        random = np.random.RandomState(seed=data_seed)
        self._load()

        if test_phase:
            self.evaluation, self.training = self._test_and_training()
        else:
            self.evaluation, self.training = self._validation_and_training(random)

        if n_labeled != 'all':
            self.training = self._unlabel(self.training, n_labeled, random)

    def _load(self):
        file_data = np.load(self.DATA_PATH)
        self._train_data = self._data_array(50000, file_data['train_x'], file_data['train_y'])
        self._test_data = self._data_array(10000, file_data['test_x'], file_data['test_y'])

    def _data_array(self, expected_n, x_data, y_data):
        array = np.zeros(expected_n, dtype=[
            ('x', np.float32, (32, 32, 3)),
            ('y', np.int32, ())  # We will be using -1 for unlabeled
        ])
        array['x'] = x_data
        array['y'] = y_data
        return array

    def _validation_and_training(self, random):
        return random_partitions(self._train_data, self.VALIDATION_SET_SIZE, random)

    def _test_and_training(self):
        return self._test_data, self._train_data

    def _unlabel(self, data, n_labeled, random):
        labeled, unlabeled = random_balanced_partitions(
            data, n_labeled, labels=data['y'], random=random)
        unlabeled['y'] = self.UNLABELED
        return np.concatenate([labeled, unlabeled])


class Eye24:

    def __init__(self, imgs_dir, train_filename, test_filename):
        self.imgs_dir = imgs_dir

        self._load(train_filename, test_filename)

    def _load_data(self, filename):
        x_data, y_data = [], []
        max_str_len = 0
        for line in open(filename, 'r'):
            iname, label = line.rstrip('\n').split(' ')
            iname = self.imgs_dir + iname
            x_data.append(iname)
            y_data.append(int(label))

            if len(iname) > max_str_len:
                max_str_len = len(iname)

        array = np.zeros(len(x_data), dtype=[
            ('x', np.str, max_str_len),
            ('y', np.int32, ())  # We will be using -1 for unlabeled
        ])
        array['x'] = x_data
        array['y'] = y_data
        return array

    def _load(self, train_filename, test_filename):
        self.training = self._load_data(train_filename)
        self.evaluation = self._load_data(test_filename)
