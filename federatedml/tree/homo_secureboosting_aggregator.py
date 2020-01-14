from federatedml.framework.homo.procedure import aggregator
from federatedml.util import consts
from functools import reduce
from arch.api.utils import log_utils
from federatedml.framework.weights import ListWeights
from federatedml.tree.feature_histogram import HistogramBag,FeatureHistogramWeights
from typing import List
import numpy as np

LOGGER = log_utils.getLogger()

class SecureBoostArbiterAggregator():

    """
     secure aggregator for secureboosting Arbiter, gather histogram and numbers
    """

    def __init__(self,transfer_variable):
        self.aggregator = aggregator.Arbiter()
        self.aggregator.register_aggregator(transfer_variable)

    def aggregate_num(self,suffix):
        self.aggregator.aggregate_loss(idx=-1,suffix=suffix)

    def aggregate_histogram(self,suffix) -> List[HistogramBag]:
        received_data = self.aggregator.get_models_for_aggregate(ciphers_dict=None,suffix=suffix)
        agg_histogram, total_degree = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), received_data)
        return agg_histogram._weights

    def broadcast_best_splits(self):
        pass


class SecureBoostClientAggregator():

    """
    secure aggregator for secureboosting Client, send histogram and numbers
    """

    def __init__(self,role,transfer_variable):
        self.aggregator = None
        if role == consts.GUEST:
            self.aggregator = aggregator.Guest()
        else:
            self.aggregator = aggregator.Host()

        self.aggregator.register_aggregator(transfer_variable)

    def send_number(self,number:float,degree:int,suffix):
        self.aggregator.send_loss(number,degree,suffix=suffix)

    def send_histogram(self,hist:List[HistogramBag],suffix):
        weights = FeatureHistogramWeights(list_of_histogrambags=hist)
        self.aggregator.send_model(weights,degree=1,suffix=suffix)

    def get_best_split_points(self):
        pass
