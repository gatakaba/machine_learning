# coding:utf-8
"""
Testing for Linear regression module.
"""
from linear_regression import LinearRegression
from numpy.testing import assert_array_almost_equal

import numpy as np
import unittest


class LinearRegressionTests(unittest.TestCase):
    def setUp(self):
        self.w = np.random.normal(size=4)
        self.learning_precision = 1
        self.generalization_precision = 1

        self.train_X = np.random.uniform(size=[100, 4])
        self.train_y = np.dot(self.train_X, self.w) + np.random.normal(size=100) * 0.01

        self.test_X = np.random.uniform(size=[100, 4])
        self.test_y = np.dot(self.test_X, self.w) + np.random.normal(size=100) * 1

        self.clf = LinearRegression()
        self.clf.fit(self.train_X, self.train_y)

    def test_learning_error(self):
        assert_array_almost_equal(self.clf.predict(self.train_X), self.train_y, decimal=self.learning_precision)

    def test_generalization_error(self):
        assert_array_almost_equal(self.clf.predict(self.test_X), self.test_y, decimal=self.generalization_precision)

    def test_lists_length(self):
        with self.assertRaises(ValueError):
            self.clf.fit([1, 2], [1, 4, 3])

    def test_empty_lists(self):
        with self.assertRaises(ValueError):
            self.clf.fit([], [])
