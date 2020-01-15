from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.feature.fate_element_type import NoneType
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.param.evaluation_param import EvaluateParam
from federatedml.util.classify_label_checker import ClassifyLabelChecker
from federatedml.util.classify_label_checker import RegressionLabelChecker
from federatedml.tree import HeteroDecisionTreeGuest
from federatedml.optim.convergence import converge_func_factory
from federatedml.tree import BoostingTree
from federatedml.tree.homo_decision_tree_arbiter import HomoDecisionTreeArbiter
from federatedml.transfer_variable.transfer_class.homo_secure_boost_transfer_variable \
    import HomoSecureBoostingTreeTransferVariable
from federatedml.transfer_variable.transfer_class.homo_decision_tree_transfer_variable import \
    HomoDecisionTreeTransferVariable
from federatedml.tree.homo_secureboosting_aggregator import SecureBoostArbiterAggregator
from federatedml.util import consts
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol import IterativeAffineEncrypt
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.loss import SigmoidBinaryCrossEntropyLoss
from federatedml.loss import SoftmaxCrossEntropyLoss
from federatedml.loss import LeastSquaredErrorLoss
from federatedml.loss import HuberLoss
from federatedml.loss import LeastAbsoluteErrorLoss
from federatedml.loss import TweedieLoss
from federatedml.loss import LogCoshLoss
from federatedml.loss import FairLoss

from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import ObjectiveMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import FeatureImportanceInfo
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from arch.api.utils import log_utils
import numpy as np
import functools
from operator import itemgetter
from numpy import random
from federatedml.util import abnormal_detection

LOGGER = log_utils.getLogger()

class HomoSecureBoostingTreeArbiter(BoostingTree):

    def __init__(self):
        super(HomoSecureBoostingTreeArbiter, self).__init__()

        self.mode = consts.HOMO
        self.feature_num = 0
        self.role = consts.ARBITER
        self.transfer_inst = HomoSecureBoostingTreeTransferVariable()


    def sample_valid_feature(self):

        chosen_feature = random.choice(range(0, self.feature_num), \
                                       max(1, int(self.subsample_feature_rate * self.feature_num)), replace=False)
        valid_features = [False for i in range(self.feature_num)]
        for fid in chosen_feature:
            valid_features[fid] = True

        return valid_features

    def sync_feature_num(self):
        feature_num_list = self.transfer_inst.feature_number.get(idx=-1,suffix=(0,))
        for num in feature_num_list[1:]:
            assert feature_num_list[0] == num
        return feature_num_list[0]

    def sync_tree_model(self):
        pass

    def fit(self, data_inst, valid_inst=None):

        self.feature_num = self.sync_feature_num()

        LOGGER.debug('begin to fit a boosting tree')
        for epoch_idx in range(self.num_trees):
            valid_feature = self.sample_valid_feature()
            new_tree = HomoDecisionTreeArbiter(self.tree_param, valid_feature=valid_feature, epoch_idx=epoch_idx,
                                               flow_id=epoch_idx)
            new_tree.fit()

        LOGGER.debug('fitting homo decision tree done')

    def predict(self, data_inst):

        LOGGER.debug('start predicting')



