from federatedml.tree.homo_secureboosting_client import HomoSecureBoostingTreeClient
from federatedml.util import consts
from arch.api.utils import log_utils
from federatedml.tree.homo_secureboosting_aggregator import SecureBoostClientAggregator
from federatedml.feature.homo_feature_binning.homo_split_points import HomoSplitPointCalculator

LOGGER = log_utils.getLogger()


class HomoSecureBoostingTreeHost(HomoSecureBoostingTreeClient):

    def __init__(self):
        super(HomoSecureBoostingTreeHost, self).__init__()
        self.role = consts.HOST
        self.aggregator = SecureBoostClientAggregator(transfer_variable=self.transfer_inst, role=self.role)
        self.binning_obj = HomoSplitPointCalculator(role=self.role, )
