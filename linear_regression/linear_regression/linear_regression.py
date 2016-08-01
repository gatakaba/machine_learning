# coding:utf-8
"""
Lienar regression model.


"""
import numpy as np


class LinearRegression(object):
    """
    Linear regression model
    y = <w, x>


    Attributes
    ----------
    w : array, shape = (n_features,)
        weight variable
    """

    def __init__(self, is_fit_intercept=False):
        """
        Parameters
        ----------
        is_fit_intercept : bool
            if is_fit_intercept is True, y = <w, x> + w_0
        """
        self.w = None
        self.is_fit_intercept = is_fit_intercept

    def fit(self, X, y):
        """
        Fit linear model.

        Parameters
        -----------
        X : numpy array of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        if not len(X) == len(y):
            raise ValueError('unequal length')

        if len(X) == 0 or len(y) == 0:
            raise ValueError('empty list')


        if self.is_fit_intercept:
            X = np.c_[np.ones(len(X)), X]

        self.w = np.dot(np.linalg.pinv(X), y)
        return self

    def predict(self, X):
        """Predict using the linear model
            Parameters
            ----------
            X : {array-like, sparse matrix}, shape = (n_samples, n_features)
                Samples.
            Returns
            -------
            y : array, shape = (n_samples,)
                Returns predicted values.
        """
        if self.is_fit_intercept:
            X = np.c_[np.ones(len(X)), X]
        y = np.dot(X, self.w)
        return y
