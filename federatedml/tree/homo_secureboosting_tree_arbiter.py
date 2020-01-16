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
from federatedml.optim.convergence import converge_func_factory
from federatedml.util import abnormal_detection
from typing import List

LOGGER = log_utils.getLogger()

class HomoSecureBoostingTreeArbiter(BoostingTree):

    def __init__(self):
        super(HomoSecureBoostingTreeArbiter, self).__init__()

        self.mode = consts.HOMO
        self.feature_num = 0
        self.role = consts.ARBITER
        self.transfer_inst = HomoSecureBoostingTreeTransferVariable()
        self.check_convergence_func = None
        self.tree_dim = None


    def sample_valid_feature(self):

        chosen_feature = random.choice(range(0, self.feature_num),\
                                       max(1, int(self.subsample_feature_rate * self.feature_num)), replace=False)
        valid_features = [False for i in range(self.feature_num)]
        for fid in chosen_feature:
            valid_features[fid] = True

        return valid_features

    def sync_feature_num(self):
        feature_num_list = self.transfer_inst.feature_number.get(idx=-1, suffix=('feat_num',))
        for num in feature_num_list[1:]:
            assert feature_num_list[0] == num
        return feature_num_list[0]

    def sync_stop_flag(self, stop_flag, suffix):
        self.transfer_inst.stop_flag.remote(stop_flag, idx=-1, suffix=suffix)

    def sync_current_loss(self, suffix):
        loss_status_list = self.transfer_inst.loss_status.get(idx=-1, suffix=suffix)
        total_loss, total_num = 0, 0
        for l_ in loss_status_list:
            total_loss += l_['cur_loss'] * l_['sample_num']
            total_num += l_['sample_num']
        LOGGER.debug('loss status received, total_loss {}, total_num {}'.format(total_loss, total_num))
        return total_loss/total_num

    def sync_tree_dim(self):
        tree_dims = self.transfer_inst.tree_dim.get(idx=-1,suffix=('tree_dim', ))
        dim0 = tree_dims[0]
        for dim in tree_dims[1:]:
            assert dim0 == dim
        return dim0

    def check_convergence(self, cur_loss):
        LOGGER.debug('checking convergence')
        return self.check_convergence_func.is_converge(cur_loss)

    def generate_flowid(self, round_num, tree_num):
        LOGGER.info("generate flowid, flowid {}".format(self.flowid))
        return ".".join(map(str, [self.flowid, round_num, tree_num]))

    def label_alignment(self) -> List:
        labels = self.transfer_inst.local_labels.get(idx=-1, suffix=('label_align', ))
        label_set = set()
        for local_label in labels:
            label_set.update(local_label)
        global_label = list(label_set)
        global_label = sorted(global_label)
        label_mapping = {v: k for k, v in enumerate(global_label)}
        self.transfer_inst.label_mapping.remote(label_mapping, idx=-1, suffix=('label_mapping', ))
        return label_mapping

    def fit(self, data_inst, valid_inst=None):

        # initializing
        self.feature_num = self.sync_feature_num()
        self.tree_dim = 1
        if self.task_type == consts.CLASSIFICATION:
            label_mapping = self.label_alignment()
            LOGGER.debug('label mapping is {}'.format(label_mapping))
            self.tree_dim = len(label_mapping) if len(label_mapping) > 2 else 1

        if self.n_iter_no_change:
            self.check_convergence_func = converge_func_factory("diff",self.tol)

        LOGGER.debug('begin to fit a boosting tree')
        for epoch_idx in range(self.num_trees):

            for t_idx in range(self.tree_dim):
                valid_feature = self.sample_valid_feature()
                flow_id = self.generate_flowid(epoch_idx, t_idx)
                new_tree = HomoDecisionTreeArbiter(self.tree_param, valid_feature=valid_feature, epoch_idx=epoch_idx,
                                                   flow_id=flow_id, tree_idx=t_idx)
                new_tree.fit()

            global_loss = self.sync_current_loss(suffix=(epoch_idx, ))
            LOGGER.debug('cur epoch global loss is {}'.format(global_loss))
            if self.n_iter_no_change:
                should_stop = self.check_convergence(cur_loss=global_loss)
                self.sync_stop_flag(should_stop, suffix=(epoch_idx, ))
                LOGGER.debug('stop flag sent')
                if should_stop:
                    break

        LOGGER.debug('fitting homo decision tree done')

    def predict(self, data_inst):

        LOGGER.debug('start predicting')



