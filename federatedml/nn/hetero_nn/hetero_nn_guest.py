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
from federatedml.optim.convergence import converge_func_factory
from federatedml.nn.hetero_nn.hetero_nn_base import HeteroNNBase
from federatedml.nn.hetero_nn.model.hetero_nn_bottom_model import HeteroNNBottomModel
from federatedml.nn.hetero_nn.model.hetero_nn_top_model import HeteroNNTopModel
from federatedml.framework.hetero.procedure import batch_generator
from federatedml.nn.hetero_nn.interactive.interactive_layer import InterActiveGuestDenseLayer
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import HeteroNNModelMeta, Optimizer
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import HeteroNNModelParam
from arch.api.utils import log_utils
from arch.api import session
import numpy as np
import json

import tensorflow as tf

LOGGER = log_utils.getLogger()
MODELMETA = "HeteroNNGuestMeta"
MODELParam = "HeteroNNGuestParam"


class HeteroNNGuest(HeteroNNBase):
    def __init__(self):
        super(HeteroNNGuest, self).__init__()
        self.task_type = None

        self.top_model = None

        self.converge_func = None

        self.batch_generator = batch_generator.Guest()
        self.data_keys = []

        self.model_builder = None
        self.label_dict = {}

        self.history_loss = []
        self.iter_epoch = 0
        self.num_label = 2

    def _init_model(self, hetero_nn_param):
        super(HeteroNNGuest, self)._init_model(hetero_nn_param)

        self.task_type = hetero_nn_param.task_type
        self.converge_func = converge_func_factory(self.early_stop, self.tol)

    def _build_bottom_model(self):
        self.bottom_model = HeteroNNBottomModel(input_shape=self.input_shape,
                                                sess=self.sess,
                                                optimizer=self.optimizer,
                                                layer_config=self.bottom_nn_define,
                                                model_builder=self.model_builder)

    def _build_top_model(self, input_shape):
        self.top_model = HeteroNNTopModel(input_shape=input_shape,
                                          sess=self.sess,
                                          optimizer=self.optimizer,
                                          layer_config=self.top_nn_define,
                                          loss=self.loss,
                                          metrics=self.metrics,
                                          model_builder=self.model_builder)

    def _build_interactive_model(self):
        self.interactive_model = InterActiveGuestDenseLayer(self.hetero_nn_param,
                                                            self.interactive_layer_define,
                                                            sess=self.sess)

        self.set_interactive_transfer_variable()

    def _build_model(self):
        self.sess = self._init_session()
        self._build_bottom_model()
        self._build_interactive_model()

        # self._build_top_model()

    def fit(self, data_inst, validate_data):
        self.prepare_batch_data(self.batch_generator, data_inst)
        self._build_model()

        while self.cur_epoch < self.epochs:
            LOGGER.debug("cur epoch is {}".format(self.cur_epoch))
            for batch_idx in range(len(self.data_x)):
                self.train_batch(self.data_x[batch_idx], self.data_y[batch_idx], self.cur_epoch, batch_idx)

            self.cur_epoch += 1

    def predict(self, data_inst):
        keys, test_x, test_y = self._load_data(data_inst)

        guest_bottom_output = self.bottom_model.predict(test_x)
        interactive_output = self.interactive_model.forward(guest_bottom_output)
        preds = self.top_model.predict(interactive_output)

        predict_tb = session.parallelize(zip(keys, preds), include_key=True)
        if self.task_type == "regression":
            result = data_inst.join(predict_tb,
                                    lambda inst, predict: [inst.label, predict[0], predict[0], {"label": predict[0]}])
        else:
            if self.num_label > 2:
                result = data_inst.join(predict_tb,
                                        lambda inst, predict: [inst.label,
                                                               np.argmax(predict),
                                                               np.max(predict),
                                                               dict([(str(idx), predict[idx]) for idx in
                                                                     range(predict.shape[0])])])

            else:
                result = data_inst.join(predict_tb,
                                        lambda inst, predict: [inst.label,
                                                               1 if predict[0] > self.predict_param.threshold else 0,
                                                               predict[0],
                                                               {"0": 1 - predict[0],
                                                                "1": predict[0]}])

        return result

    def train_batch(self, x, y, epoch, batch_idx):
        guest_bottom_output = self.bottom_model.forward(x)
        interactive_output = self.interactive_model.forward(guest_bottom_output, epoch, batch_idx)

        if self.top_model is None:
            self._build_top_model(interactive_output.shape[1:])

        gradients = self.top_model.train_and_get_backward_gradient(interactive_output, y)

        guest_backward = self.interactive_model.backward(gradients, epoch, batch_idx)
        self.bottom_model.backward(x, guest_backward)

    def export_model(self):
        return {MODELMETA: self._get_model_meta(),
                MODELParam: self._get_model_param()}

    def _get_model_meta(self):
        model_meta = HeteroNNModelMeta()
        model_meta.task_type = self.task_type
        model_meta.config_type = self.config_type
        model_meta.loss = self.loss

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

        optimizer = Optimizer()
        if isinstance(self.optimizer, str):
            optimizer.optimizer = self.optimizer
        else:
            optimizer.args = json.dumps(self.optimizer)

        model_meta.optimizer = optimizer
        for loss in self.history_loss:
            model_meta.append(loss)

    def _get_model_param(self):
        model_param = HeteroNNModelParam()
        model_param.iter_epoch = self.iter_epoch
        model_param.bottom_saved_model_bytes = self.bottom_model.export_model()
        model_param.top_saved_model_bytes = self.top_model.export_model()
        model_param.interactive_save_model_bytes = self.interactive_model.saved_model_bytes()

    def prepare_batch_data(self, batch_generator, data_inst):
        batch_generator.initialize_batch_generator(data_inst, self.batch_size)
        batch_data_generator = batch_generator.generate_batch_data()

        for batch_data in batch_data_generator:
            keys, batch_x, batch_y = self._load_data(batch_data)
            self.data_x.append(batch_x)
            self.data_y.append(batch_y)
            self.data_keys.append(keys)

        self._convert_label()

    def _load_data(self, data_inst):
        keys = []
        batch_x = []
        batch_y = []
        for key, inst in data_inst.collect():
            keys.append(key)
            batch_x.append(inst.features)
            batch_y.append(inst.label)

            if self.input_shape is None:
                self.input_shape = inst.features.shape

        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)

        return keys, batch_x, batch_y

    def _convert_label(self):
        diff_label = np.unique([batch_y for batch_y in self.data_y])
        self.label_dict = dict(zip(diff_label, range(diff_label.shape[0])))

        transform_y = []
        self.num_label = diff_label.shape[0]

        if self.num_label <= 2:
            for batch_y in self.data_y:
                new_batch_y = np.zeros((batch_y.shape[0], 1))
                for idx in range(new_batch_y.shape[0]):
                    new_batch_y[idx][0] = batch_y[idx]

                transform_y.append(new_batch_y)

            self.data_y = transform_y
            return

        for batch_y in self.data_y:
            new_batch_y = np.zeros((batch_y.shape[0], self.num_label))
            for idx in range(new_batch_y.shape[0]):
                y = new_batch_y[idx]
                new_batch_y[idx][y] = 1

            transform_y.append(new_batch_y)

        self.data_y = transform_y
