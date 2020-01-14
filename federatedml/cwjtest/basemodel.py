import functools
import copy
from arch.api import session
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import CriterionMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import DecisionTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import DecisionTreeModelParam
from federatedml.transfer_variable.transfer_class.hetero_decision_tree_transfer_variable import \
    HeteroDecisionTreeTransferVariable
from federatedml.util import consts
from federatedml.tree import FeatureHistogram
from federatedml.tree import DecisionTree
from federatedml.tree import Splitter
from federatedml.tree import Node
from federatedml.feature.fate_element_type import NoneType
from federatedml.model_base import ModelBase
from federatedml.transfer_variable.transfer_class.cwjtest_transfer_variable import CWJTestVariable
from federatedml.param.cwjtestparam import CwjParam
from federatedml.framework.homo.procedure import aggregator
from federatedml.tree.feature_histogram import HistogramBag
from arch.api.utils import log_utils
from federatedml.framework.weights import NumericWeights
from federatedml.framework.weights import ListWeights


LOGGER = log_utils.getLogger()

class CWJBase(ModelBase):

    def __init__(self):
        super(CWJBase,self).__init__()
        self.tranvar = CWJTestVariable()
        self.model_param = CwjParam()
        LOGGER.debug('initializing aggregator')
        self.aggregator = None
        self.hist = None
        self.list_weights = None


    def _init_model(self,param):
        LOGGER.info('initialization done')

    def fit(self,train=None,validate=None):
        LOGGER.debug('fit called')
        self.aggregator.send_histogram([self.hist,self.hist],suffix=('cwj',))
        # data = self.sync_data_vara()
        LOGGER.debug(self.hist.flatten())
        LOGGER.info('data sent')
