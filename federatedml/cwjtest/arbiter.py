from arch.api.utils import log_utils


from federatedml.util import consts
from federatedml.model_base import ModelBase
from federatedml.transfer_variable.transfer_class.cwjtest_transfer_variable import CWJTestVariable
from federatedml.param.cwjtestparam import CwjParam
from federatedml.framework.homo.procedure import aggregator
from federatedml.tree.homo_secureboosting_aggregator import SecureBoostArbiterAggregator
from federatedml.framework.weights import Weights,DictWeights

LOGGER = log_utils.getLogger()

class CWJArbiter(ModelBase):

    def __init__(self):
        super(CWJArbiter, self).__init__()
        self.role = consts.ARBITER
        self.model_param = CwjParam()
        self.tranvar = CWJTestVariable()

        LOGGER.debug('initializing aggregator')
        self.aggregator = SecureBoostArbiterAggregator(transfer_variable=self.tranvar)

    def fit(self,train=None,validate=None):
        LOGGER.debug('fit called')
        agg = self.aggregator.aggregate_histogram(suffix=('cwj',))
        for bag in agg:
            LOGGER.debug(bag)
        LOGGER.info('got data')
        weights = DictWeights({'greeting':'nmsl','g':123,'h':114514.12})
        root_info = self.aggregator.aggregate_root_node_info(suffix=('cwj2',))
        LOGGER.debug('root info:')
        LOGGER.debug(root_info)
        self.aggregator.broadcast_root_info(root_info[0],root_info[1],root_info[2],
                                            suffix=('cwj1',))

    def _init_model(self,param):
        LOGGER.info('initialization done')
