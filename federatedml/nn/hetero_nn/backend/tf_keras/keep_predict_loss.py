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

from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import keras_export

"""
@keras_export('keras.losses.KeepPredictLoss')
class KeepPredictLoss(LossFunctionWrapper):
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name="keep_predict_loss"):
        super(KeepPredictLoss, self).__init__(
            keep_predict_loss, name=name, reduction=reduction)
"""


@keras_export('keras.losses.keep_predict_loss')
def keep_predict_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_true * y_pred)
    return K.mean(y_true * y_pred)


"""
import keras.backend as K


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
"""
