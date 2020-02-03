#!/usr/bin/env python    
# -*- coding: utf-8 -*- 

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
################################################################################
#
#
################################################################################

# =============================================================================
# Decision Tree Node Struture
# =============================================================================

from federatedml.util import consts
from federatedml.tree.feature_histogram import HistogramBag

class Node(object):
    def __init__(self, id=None, sitename=consts.GUEST, fid=None,
                 bid=None, weight=0, is_leaf=False, sum_grad=None,
                 sum_hess=None, left_nodeid=-1, right_nodeid=-1,
                 missing_dir=1, sample_num=0, sibling_nodeid=None, parent_nodeid=None, is_left_node=False):
        self.id = id
        self.sitename = sitename
        self.fid = fid
        self.bid = bid
        self.weight = weight
        self.is_leaf = is_leaf
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess
        self.left_nodeid = left_nodeid
        self.right_nodeid = right_nodeid
        self.missing_dir = missing_dir
        self.stored_histogram = None
        self.sibling_nodeid = sibling_nodeid
        self.parent_nodeid = parent_nodeid
        self.sample_num = sample_num
        self.is_left_node = is_left_node

    def set_histogram(self, hist: HistogramBag):
        self.stored_histogram = hist

    def __str__(self):
        return "fid:{},bid:{},weight:{},sum_grad:{},sum_hess:{},left_node:{},right_node:{}".format(
                self.fid, self.bid, self.weight, self.sum_grad, self.sum_hess, self.left_nodeid, self.right_nodeid
        )


class SplitInfo(object):
    def __init__(self, sitename=consts.GUEST, best_fid=None, best_bid=None,
                 sum_grad=0, sum_hess=0, gain=None, missing_dir=1, left_sample_count=0):
        self.sitename = sitename
        self.best_fid = best_fid
        self.best_bid = best_bid
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess
        self.gain = gain
        self.missing_dir = missing_dir

    def __str__(self):
        return 'best_fid:{},best_bid{},sum_grad{},sum_hess{},gain{}'.format(self.best_fid, self.best_bid,
                                                                            self.sum_grad, self.sum_hess, self.gain)
