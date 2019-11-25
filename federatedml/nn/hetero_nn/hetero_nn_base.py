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
from federatedml.model_base import ModelBase
from tensorflow.python.keras.backend import set_session
from federatedml.nn.homo_nn import nn_model
from federatedml.transfer_variable.transfer_class.hetero_nn_transfer_variable import HeteroNNTransferVariable
from federatedml.param.hetero_nn_param import HeteroNNParam
import tensorflow as tf
from types import SimpleNamespace


class HeteroNNBase(ModelBase):
    def __init__(self):
        super(HeteroNNBase, self).__init__()

        self.config_type = "keras"
        self.bottom_nn_define = None
        self.top_nn_define = None
        self.interactive_layer_define = None

        self.bottom_model = None
        self.interactive_model = None

        self.optimizer = None

        self.tol = None
        self.early_stop = None

        self.epochs = None
        self.batch_size = None

        self.loss = None
        self.metrics = None

        self.predict_param = None
        self.hetero_nn_param = None

        self.cur_epoch = 0

        self.input_shape = None
        self.model_builder = None

        self.sess = None
        self.data_x = []
        self.data_y = []
        self.transfer_variable = HeteroNNTransferVariable()

        self.model_param = HeteroNNParam()

    def _init_model(self, hetero_nn_param):
        self.config_type = hetero_nn_param.config_type

        self.bottom_nn_define = hetero_nn_param.bottom_nn_define

        self.top_nn_define = hetero_nn_param.top_nn_define
        self.interactive_layer_define = hetero_nn_param.interactive_layer_define
        self.interactive_layer_lr = hetero_nn_param.interactive_layer_lr

        self.optimizer = self._parse_optimizer(hetero_nn_param.optimizer)
        self.epochs = hetero_nn_param.epochs
        self.batch_size = hetero_nn_param.batch_size

        self.early_stop = hetero_nn_param.early_stop
        self.tol = hetero_nn_param.tol

        self.loss = hetero_nn_param.loss
        self.metrics = hetero_nn_param.metrics

        self.predict_param = hetero_nn_param.predict_param
        self.hetero_nn_param = hetero_nn_param

        self.model_builder = nn_model.get_nn_builder(config_type=self.config_type)

        self.batch_generator.register_batch_generator(self.transfer_variable)

    def _build_bottom_model(self):
        pass

    def _build_interactive_model(self):
        pass

    def set_interactive_transfer_variable(self):
        self.interactive_model.set_transfer_variable(self.transfer_variable)

    def prepare_batch_data(self, batch_generator, data_inst):
        pass

    def _load_data(self, data_inst):
        pass

    @staticmethod
    def _init_session():
        sess = tf.Session()
        tf.get_default_graph()
        set_session(sess)
        return sess

    @staticmethod
    def _parse_optimizer(opt):
        """
        Examples:

            1. "optimize": "SGD"
            2. "optimize": {
                "optimizer": "SGD",
                "learning_rate": 0.05
            }
        """

        kwargs = {}
        if isinstance(opt, str):
            return SimpleNamespace(optimizer=opt, kwargs=kwargs)
        elif isinstance(opt, dict):
            optimizer = opt.get("optimizer", kwargs)
            if not optimizer:
                raise ValueError(f"optimizer config: {opt} invalid")
            kwargs = {k: v for k, v in opt.items() if k != "optimizer"}
            return SimpleNamespace(optimizer=optimizer, kwargs=kwargs)
        else:
            raise ValueError(f"invalid type for optimize: {type(opt)}")
