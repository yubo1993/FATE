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

from federatedml.nn.hetero_nn.losses import keep_predict_loss
from federatedml.nn.hetero_nn.util.data_generator import KerasSequenceData


class HeteroNNBottomModel(object):
    def __init__(self, input_shape=None, sess=None, optimizer="SGD", model_builder=None, layer_config=None):
        loss = "keep_predict_loss"
        self._keras_model = model_builder(input_shape=input_shape,
                                          nn_define=layer_config,
                                          optimizer=optimizer,
                                          loss=loss,
                                          metrics=None,
                                          sess=sess)

        self.model = self._keras_model._model

    def get_output_shape(self):
        return self.model.fit()

    def forward(self, x):
        seq_data = KerasSequenceData(x)
        output_data = self._keras_model.predict(seq_data)

        return output_data

    def backward(self, x, y):
        seq_data = KerasSequenceData(x, y)
        self._keras_model.train(seq_data)

    def predict(self, x):
        seq_data = KerasSequenceData(x)

        return self._keras_model.predict(seq_data)

    def export_model(self):
        return self._keras_model.export_model()

    def restore_mode(self, sess, model_bytes):
        self._keras_model = self._keras_model.restore_model(model_bytes, sess)
