from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from ..base import ModelingMixin, BaseEstimator

import numpy as np
from scipy.optimize import curve_fit

class NonLinearModel(LinearModel):
    pass

class NonLinearRegression(ModelingMixin, NonLinearModel):
    def __init__(self, nl_func, initial_params=None, bounds=None):
        """
        :param nl_func: callable, args = (X, *params). Note that X must be (arg number, n)
        """
        self.nl_func = nl_func
        self.initial_params = initial_params
        self.bounds = bounds if bounds else (-np.inf, np.inf)

    def fit(self, X, y):
        """
        fitting with nonlinear regression
        :param X: 2d ndarray, shape = (n,*=arg number)
        :param y: 1d ndarray, shape = (n,)
        """
        # Check X and y have (n,*) and (n,) respectively
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        popt, pcov = curve_fit(self.nl_func, np.swapaxes(X, 0, 1), y, p0=self.initial_params, bounds=self.bounds)
        self.coefs_ = np.array(popt)  # shape = (params num,)

        return self

    def predict(self, X):
        # Check this estimator had been fitted
        check_is_fitted(self, attributes=('coefs_'))

        # Check X is correct
        X = check_array(X)

        y_pred = self.nl_func(np.swapaxes(X, 0, 1), *self.coefs_)
        return np.array(y_pred)


class PDFEstimator(ModelingMixin, BaseEstimator):
    def __init__(self, estimation_methods='mle'):
        """
        :param estimation_methods: estimation methods.
                mle: maximum likelihood estimation
                nlr: non linear regression
        """
        pass

    def fit(self, X, y):
        """
        fitting with nonlinear regression
        :param X: Random variable. 2d ndarray, shape = (n,*=arg number)
        :param y: Probability. 1d ndarray, shape = (n,)
        """
        pass

class PDFMLEMixin:
    def fit_mle(self):
        pass