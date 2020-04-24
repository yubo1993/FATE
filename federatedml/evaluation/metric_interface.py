from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import numpy as np
from federatedml.util import consts
import logging
from arch.api.utils import log_utils

from federatedml.evaluation.metrics import classification_metric
from federatedml.evaluation.metrics import regression_metric

from functools import wraps

LOGGER = log_utils.getLogger()


class Metrics(object):

    def __init__(self, pos_label: int, eval_type: str):

        self.pos_label = pos_label
        self.eval_type = eval_type

    def auc(self, labels, pred_scores):
        """
        Compute AUC for binary classification.

        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.

        Returns
        ----------
        float
            The AUC
        """
        if self.eval_type == consts.BINARY:
            return roc_auc_score(labels, pred_scores)
        else:
            logging.warning("auc is just suppose Binary Classification! return None as results")
            return None

    @staticmethod
    def explained_variance(labels, pred_scores):
        """
        Compute explain variance
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.

        Returns
        ----------
        float
            The explain variance
        """
        return regression_metric.ExplainedVariance().compute(labels, pred_scores)

    @staticmethod
    def mean_absolute_error(labels, pred_scores):
        """
        Compute mean absolute error
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        float
            A non-negative floating point.
        """
        return regression_metric.MAE().compute(labels, pred_scores)

    @staticmethod
    def mean_squared_error(labels, pred_scores):
        """
        Compute mean square error
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        float
            A non-negative floating point value
        """
        return regression_metric.MSE.compute(labels, pred_scores)

    @staticmethod
    def median_absolute_error(labels, pred_scores):
        """
        Compute median absolute error
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        float
            A positive floating point value
        """
        return regression_metric.MedianAbsoluteError().compute(labels, pred_scores)

    @staticmethod
    def r2_score(labels, pred_scores):
        """
        Compute R^2 (coefficient of determination) score
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        float
            The R^2 score
        """
        return regression_metric.R2Score().compute(labels, pred_scores)

    @staticmethod
    def root_mean_squared_error(labels, pred_scores):
        """
        Compute the root of mean square error
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Return
        ----------
        float
            A positive floating point value
        """
        return regression_metric.RMSE.compute(labels, pred_scores)

    @staticmethod
    def __filt_threshold(thresholds, step):
        cuts = list(map(float, np.arange(0, 1, step)))
        size = len(list(thresholds))
        thresholds.sort(reverse=True)
        index_list = [int(size * cut) for cut in cuts]
        new_thresholds = [thresholds[idx] for idx in index_list]

        return new_thresholds, cuts

    def roc(self, labels, pred_scores):
        if self.eval_type == consts.BINARY:
            fpr, tpr, thresholds = roc_curve(np.array(labels), np.array(pred_scores), drop_intermediate=1)
            fpr, tpr, thresholds = list(map(float, fpr)), list(map(float, tpr)), list(map(float, thresholds))

            filt_thresholds, cuts = self.__filt_threshold(thresholds=thresholds, step=0.01)
            new_thresholds = []
            new_tpr = []
            new_fpr = []
            for threshold in filt_thresholds:
                index = thresholds.index(threshold)
                new_tpr.append(tpr[index])
                new_fpr.append(fpr[index])
                new_thresholds.append(threshold)

            fpr = new_fpr
            tpr = new_tpr
            thresholds = new_thresholds
            return fpr, tpr, thresholds, cuts
        else:
            logging.warning("roc_curve is just suppose Binary Classification! return None as results")
            fpr, tpr, thresholds, cuts = None, None, None, None

            return fpr, tpr, thresholds, cuts

    def ks(self, labels, pred_scores):

        """
        Compute Kolmogorov-Smirnov
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        max_ks_interval: float max value of each tpr - fpt
        fpr:
        """

        return classification_metric.KS().compute(labels, pred_scores)

    def lift(self, labels, pred_scores):
        """
        Compute lift of binary classification.
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        thresholds: value list. This parameter effective only for 'binary'. The predict scores will be 1 if it larger than thresholds, if not,
                    if will be 0. If not only one threshold in it, it will return several results according to the thresholds. default None
        Returns
        ----------
        float
            The lift
        """
        if self.eval_type == consts.BINARY:
            return classification_metric.Lift().compute(labels, pred_scores)
        else:
            logging.warning("lift is just suppose Binary Classification! return None as results")
            return None

    def gain(self, labels, pred_scores):
        """
        Compute gain of binary classification.
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        thresholds: value list. This parameter effective only for 'binary'. The predict scores will be 1 if it larger than thresholds, if not,
                    if will be 0. If not only one threshold in it, it will return several results according to the thresholds. default None
        Returns
        ----------
        float
            The gain
        """

        if self.eval_type == consts.BINARY:
            return classification_metric.Gain().compute(labels, pred_scores)
        else:
            logging.warning("gain is just suppose Binary Classification! return None as results")
            return None

    def precision(self, labels, pred_scores):
        """
        Compute the precision
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        thresholds: value list. This parameter effective only for 'binary'. The predict scores will be 1 if it larger than thresholds, if not,
                    if will be 0. If not only one threshold in it, it will return several results according to the thresholds. default None
        result_filter: value list. If result_filter is not None, it will filter the label results not in result_filter.
        Returns
        ----------
        dict
            The key is threshold and the value is another dic, which key is label in parameter labels, and value is the label's precision.
        """
        if self.eval_type == consts.BINARY:
            precision_operator = classification_metric.BiClassPrecision()
            return precision_operator.compute(labels, pred_scores)
        elif self.eval_type == consts.MULTY:
            precision_operator = classification_metric.MultiClassPrecision()
            return precision_operator.compute(labels, pred_scores)
        else:
            logging.warning("error:can not find classification type:{}".format(self.eval_type))

    def recall(self, labels, pred_scores):
        """
        Compute the recall
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        dict
            The key is threshold and the value is another dic, which key is label in parameter labels, and value is the label's recall.
        """
        if self.eval_type == consts.BINARY:
            recall_operator = classification_metric.BiClassRecall()
            recall_res, thresholds, cuts = recall_operator.compute(labels, pred_scores)
            return recall_res, thresholds, cuts
        elif self.eval_type == consts.MULTY:
            recall_operator = classification_metric.MultiClassRecall()
            return recall_operator.compute(labels, pred_scores)
        else:
            logging.warning("error:can not find classification type:{}".format(self.eval_type))

    def accuracy(self, labels, pred_scores, normalize=True):
        """
        Compute the accuracy
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        normalize: bool. If true, return the fraction of correctly classified samples, else returns the number of correctly classified samples
        Returns
        ----------
        dict
            the key is threshold and the value is the accuracy of this threshold.
        """

        if self.eval_type == consts.BINARY:
            acc_operator = classification_metric.BiClassAccuracy()
            acc_res, thresholds, cuts = acc_operator.compute(labels, pred_scores, normalize)
            return acc_res, cuts, thresholds
        elif self.eval_type == consts.MULTY:
            acc_operator = classification_metric.MultiClassAccuracy()
            return acc_operator.compute(labels, pred_scores, normalize)
        else:
            logging.warning("error:can not find classification type:".format(self.eval_type))