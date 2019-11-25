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


from federatedml.nn.hetero_nn.hetero_nn_base import HeteroNNBase
from federatedml.nn.hetero_nn.model.hetero_nn_bottom_model import HeteroNNBottomModel
from federatedml.nn.hetero_nn.interactive.interactive_layer import InteractiveHostDenseLayer
from federatedml.framework.hetero.procedure import batch_generator
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import HeteroNNModelMeta, Optimizer
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import HeteroNNModelParam

import numpy as np


MODELMETA = "HeteroNNHostMeta"
MODELParam= "HeteroNNGHostParam"


class HeteroNNHost(HeteroNNBase):
    def __init__(self):
        super(HeteroNNHost, self).__init__()

        self.batch_generator = batch_generator.Host()

    def _init_model(self, hetero_nn_param):
        super(HeteroNNHost, self)._init_model(hetero_nn_param)

    def _load_model(self, model_dict):
        pass

    def predict(self, data_inst):
        test_x = self._load_data(data_inst)

        guest_bottom_output = self.bottom_model.predict(test_x)
        self.interactive_model.forward(guest_bottom_output)

    def _build_bottom_model(self):
        self.bottom_model = HeteroNNBottomModel(input_shape=self.input_shape,
                                                sess=self.sess,
                                                optimizer=self.optimizer,
                                                layer_config=self.bottom_nn_define,
                                                model_builder=self.model_builder)

    def _build_interactive_model(self):
        self.interactive_model = InteractiveHostDenseLayer(self.hetero_nn_param)
        self.set_interactive_transfer_variable()

    def _build_model(self):
        self.sess = self._init_session()
        self._build_bottom_model()
        self._build_interactive_model()

    def fit(self, data_inst, validate_data):
        self.prepare_batch_data(self.batch_generator, data_inst)
        self._build_model()

        while self.cur_epoch < self.epochs:
            for batch_idx in range(len(self.data_x)):
                self.train_batch(self.data_x[batch_idx], self.cur_epoch, batch_idx)

            self.cur_epoch += 1

    def prepare_batch_data(self, batch_generator, data_inst):
        batch_generator.initialize_batch_generator(data_inst)
        batch_data_generator = batch_generator.generate_batch_data()

        for batch_data in batch_data_generator:
            batch_x = self._load_data(batch_data)
            self.data_x.append(batch_x)

    def train_batch(self, x, epoch, batch_idx):
        host_bottom_output = self.bottom_model.forward(x)
        self.interactive_model.forward(host_bottom_output, epoch, batch_idx)

        host_gradient = self.interactive_model.backward(epoch, batch_idx)
        self.bottom_model.backward(x, host_gradient)

    def _load_data(self, data_inst):
        batch_x = []
        for key, inst in data_inst.collect():
            batch_x.append(inst.features)

            if self.input_shape is None:
                self.input_shape = inst.features.shape

        batch_x = np.asarray(batch_x)

        return batch_x

    def _get_model_meta(self):
        model_meta = HeteroNNModelMeta()
        model_meta.config_type = self.config_type

        if self.config_type == "nn":
            for layer in self.bottom_nn_define:
                model_meta.bottom_nn_define.append(json.dumps(layer))

            for layer in self.top_nn_define:
                model_meta.top_nn_define.append(json.dumps(layer))
        elif self.config_type == "keras":
            model_meta.bottom_nn_define.append(json.dumps(self.bottom_nn_define))
            model_meta.top_nn_define.append(json.dumps(self.top_nn_define))

        model_meta.interactive_layer_define = json.dumps(self.interactive_layer_define)
        model_meta.batch_size = self.batch_size
        model_meta.epochs = self.epochs
        model_meta.early_stop = self.early_stop
        model_meta.tol = self.tol

        for metric in self.metrics:
            model_meta.metrics.append(metric)

    def _get_model_param(self):
        model_param = HeteroNNModelParam()
        model_param.bottom_saved_model_bytes = self.bottom_model.export_model()
        model_param.interactive_save_model_bytes = self.interactive_model.saved_model_bytes()


