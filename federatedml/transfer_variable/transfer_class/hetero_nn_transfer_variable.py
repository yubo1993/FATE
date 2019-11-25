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
# AUTO GENERATED TRANSFER VARIABLE CLASS. DO NOT MODIFY
#
################################################################################

from federatedml.transfer_variable.transfer_class.base_transfer_variable import BaseTransferVariable, Variable


# noinspection PyAttributeOutsideInit
class HeteroNNTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.encrypted_host_forward = Variable(name='HeteroNNTransferVariable.encrypted_host_forward', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.encrypted_guest_forward = Variable(name='HeteroNNTransferVariable.encrypted_guest_forward', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.decrypted_guest_fowrad = Variable(name='HeteroNNTransferVariable.decrypted_guest_fowrad', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.encrypted_guest_weight_gradient = Variable(name='HeteroNNTransferVariable.encrypted_guest_weight_gradient', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.decrypted_guest_weight_gradient = Variable(name='HeteroNNTransferVariable.decrypted_guest_weight_gradient', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.host_backward = Variable(name='HeteroNNTransferVariable.host_backward', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.batch_data_index = Variable(name='HeteroNNTransferVariable.batch_data_index', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.batch_info = Variable(name='HeteroNNTransferVariable.batch_info', auth=dict(src='guest', dst=['host', 'arbiter']), transfer_variable=self)
        pass
