from federatedml.tree.homo_secureboosting_client import HomoSecureBoostingTreeClient
from federatedml.tree.homo_secureboosting_aggregator import SecureBoostClientAggregator
from federatedml.util import consts
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HomoSecureBoostingTreeHost(HomoSecureBoostingTreeClient):

    def __init__(self):
        super(HomoSecureBoostingTreeHost, self).__init__()
        self.role = consts.HOST
