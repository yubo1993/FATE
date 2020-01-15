from federatedml.framework.homo.procedure import aggregator
from federatedml.util import consts
from functools import reduce
from arch.api.utils import log_utils
from federatedml.framework.weights import DictWeights
from federatedml.tree.feature_histogram import HistogramBag,FeatureHistogramWeights
from typing import List
import numpy as np
from typing import Dict

LOGGER = log_utils.getLogger()

class SecureBoostArbiterAggregator():

    """
     secure aggregator for secureboosting Arbiter, gather histogram and numbers
    """

    def __init__(self,transfer_variable):
        self.aggregator = aggregator.Arbiter()
        self.aggregator.register_aggregator(transfer_variable,enable_secure_aggregate=True)

    def aggregate_num(self,suffix):
        self.aggregator.aggregate_loss(idx=-1,suffix=suffix)

    def aggregate_histogram(self,suffix) -> List[HistogramBag]:
        received_data = self.aggregator.get_models_for_aggregate(ciphers_dict=None,suffix=suffix)
        LOGGER.debug('showing received data')
        def reduce_func(x,y):
            LOGGER.debug('guest')
            LOGGER.debug(x[0]._weights[0])
            LOGGER.debug('host')
            LOGGER.debug(y[0]._weights[0])
            return x[0]+y[0],x[1]+y[1]
        agg_histogram, total_degree = reduce(reduce_func, received_data)
        return agg_histogram._weights

    def broadcast_best_splits(self):
        pass

    def aggregate_root_node_info(self,suffix):

        data = self.aggregator.get_models_for_aggregate(ciphers_dict=None, suffix=suffix)
        agg_data, total_degree = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), data)
        d = agg_data._weights
        return d['g_sum'],d['h_sum']

    def broadcast_root_info(self,g_sum,h_sum,suffix):
        d = {'g_sum':g_sum,'h_sum':h_sum}
        weight = DictWeights(d=d,)
        self.aggregator.send_aggregated_model(weight,suffix=suffix)


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

        self.aggregator.register_aggregator(transfer_variable,enable_secure_aggregate=True)

    def send_number(self,number:float,degree:int,suffix):
        self.aggregator.send_loss(number,degree,suffix=suffix)

    def send_histogram(self,hist:List[HistogramBag],suffix):
        for idx, histbag in enumerate(hist):
            LOGGER.debug('showing client hist {}'.format(idx))
            LOGGER.debug(histbag)
        weights = FeatureHistogramWeights(list_of_histogrambags=hist)
        self.aggregator.send_model(weights,degree=1,suffix=suffix)

    def get_aggregated_root_info(self,suffix) -> Dict:
        dict_weight = self.aggregator.get_aggregated_model(suffix=suffix)
        content = dict_weight._weights
        return content

    def send_local_root_node_info(self,g_sum,h_sum,suffix):
        d = {'g_sum':g_sum,'h_sum':h_sum,}
        dict_weights = DictWeights(d=d)
        self.aggregator.send_model(dict_weights,suffix=suffix)

    def get_best_split_points(self):
        pass
