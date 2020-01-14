from federatedml.cwjtest.basemodel import CWJBase
from federatedml.util import consts
from federatedml.framework.homo.procedure import aggregator
from federatedml.framework.weights import NumericWeights
from federatedml.tree.feature_histogram import HistogramBag
from federatedml.tree.homo_secureboosting_aggregator import SecureBoostArbiterAggregator,SecureBoostClientAggregator
from federatedml.framework.weights import ListWeights

class CWJGuest(CWJBase):

    def __init__(self):
        super(CWJGuest,self).__init__()

        self.role = consts.GUEST
        self.aggregator = SecureBoostClientAggregator(role=self.role,transfer_variable=self.tranvar)

        self.hist = HistogramBag(2,[1,1])
        self.hist[0][0][1] = 123
        self.hist[1][0][0] = 114

        self.list_weights = ListWeights([3,2,1])

