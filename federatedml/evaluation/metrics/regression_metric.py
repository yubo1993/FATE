from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

import numpy as np


class RMSE(object):

    @staticmethod
    def compute(labels, pred_scores):
        return np.sqrt(mean_squared_error(labels, pred_scores))


class MAE(object):

    @staticmethod
    def compute(labels, pred_scores):
        return mean_absolute_error(labels, pred_scores)


class R2Score(object):

    @staticmethod
    def compute(labels, pred_scores):
        return r2_score(labels, pred_scores)


class MSE(object):

    @staticmethod
    def compute(labels, pred_scores):
        return mean_squared_error(labels, pred_scores)


class ExplainedVariance(object):

    @staticmethod
    def compute(labels, pred_scores):
        return explained_variance_score(labels, pred_scores)


class MedianAbsoluteError(object):

    @staticmethod
    def compute(labels, pred_scores):
        return median_absolute_error(labels, pred_scores)