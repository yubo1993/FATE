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
from federatedml.nn.homo_nn.zoo.nn import build_nn_model
from federatedml.nn.homo_nn.backend.tf_keras.nn_model import KerasNNModel
from federatedml.nn.hetero_nn.util.data_generator import KerasSequenceData


class HeteroNNTopModel(object):
    def __init__(self, input_shape=None, sess=None, loss=None, optimizer="SGD", metrics=None, model_builder=None, layer_config=None):
        self._keras_model = model_builder(input_shape=input_shape,
                                          nn_define=layer_config,
                                          optimizer=optimizer,
                                          loss=loss,
                                          metrics=metrics,
                                          sess=sess)


        self.gradient = None

    def train_and_get_backward_gradient(self, x, y):
        gradients = self._keras_model.get_gradients(x, y)

        seq_data = KerasSequenceData(x, y)
        self._keras_model.train(seq_data)

        return gradients

    """
    def train(self, input_data):
        gradients = self._keras_model.get_gradients(input_data.X, input_data.y)
        self._keras_model.train(input_data)

        return gradients
    """

    def predict(self, input_data):
        output_data = self._keras_model.predict(input_data)

        return output_data

    def export_model(self):
        return self._keras_model.export_model()

    def restore_mode(self, sess, model_bytes):
        self._keras_model = self._keras_model.restore_model(model_bytes, sess)
