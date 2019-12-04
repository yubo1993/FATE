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

from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.backend import gradients

from arch.api.utils import log_utils
from federatedml.nn.hetero_nn.backend.ops import HeteroNNTensor
from federatedml.nn.hetero_nn.backend.tf_keras.losses import keep_predict_loss
from federatedml.nn.hetero_nn.util import random_number_generator
from federatedml.nn.homo_nn.backend.tf_keras.layers import get_builder
from federatedml.secureprotol import PaillierEncrypt
from federatedml.util import consts

LOGGER = log_utils.getLogger()

"""
TODO:
      1. make Guest\Host a single object, to support guest' input_shape = 0
      2. Parallelize
      3. change Acc_noise to vector 
      4  backend abstract
"""


def _build_model(input_shape, layer_config, sess):
    model = Sequential()

    if "layer" in layer_config:
        del layer_config["layer"]
    layer_config["input_shape"] = input_shape
    builder = get_builder("Dense")
    model.add(builder(**layer_config["config"]))
    model.compile("SGD", loss=keep_predict_loss)

    sess.run(tf.initialize_all_variables())

    return model


class DenseModel(object):
    def __init__(self):
        self.input = None
        self.input_shape = None
        self.model_weight = None
        self.model_shape = None
        self.bias = None
        self.model = None
        self.input = None
        self.lr = 1.0
        self.layer_config = None
        self.sess = None
        self.role = "host"
        self.activation_placeholder_name = "activation_placeholder"
        self.activation_gradient_func = None
        self.activation_func = None
        self.is_empty_model = False
        self.activation_input = None

    def forward_dense(self, x):
        pass

    def apply_update(self, delta):
        pass

    def get_gradient(self, delta):
        pass

    """
    def get_backward_gradient(self, delta, **kwargs):
        pass
    """

    def set_sess(self, sess):
        self.sess = sess

    def build(self, input_shape, layer_config, sess):
        if not input_shape:
            if self.role == "host":
                raise ValueError("host input is empty!")
            else:
                self.is_empty_model = True
                return

        self.sess = sess

        self.layer_config = layer_config
        self.input_shape = input_shape
        self.model = _build_model(input_shape, layer_config, self.sess)
        trainable_weights = self.sess.run(self.model.trainable_weights)
        self.model_weight = trainable_weights[0]
        self.model_shape = self.model.get_weights()[0].shape

        if self.role == "host":
            if self.model.layers[0].use_bias:
                self.bias = trainable_weights[1]

            self.activation_func = self.model.layers[0].activation
            self.__build_activation_layer_gradients_func(self.model)

    def __build_activation_layer_gradients_func(self, model):
        dense_layer = model.layers[0]
        shape = dense_layer.output_shape
        dtype = dense_layer.get_weights()[0].dtype

        input_data = tf.placeholder(shape=shape,
                                    dtype=dtype,
                                    name=self.activation_placeholder_name)

        self.activation_gradient_func = gradients(dense_layer.activation(input_data), input_data)

    def forward_activation(self, input_data):
        self.activation_input = input_data

        output = self.activation_func(input_data)
        if not isinstance(output, np.ndarray):
            output =  self.sess.run(output)

        return output

    def backward_activation(self):
        placeholder = tf.get_default_graph().get_tensor_by_name(":".join([self.activation_placeholder_name, "0"]))
        return self.sess.run(self.activation_gradient_func,
                             feed_dict={placeholder: self.activation_input})

    def get_weight(self):
        return self.model_weight

    def get_bias(self):
        return self.bias

    def get_input_shape(self):
        return self.input_shape

    @property
    def empty(self):
        return self.is_empty_model

    @property
    def output_shape(self):
        return self.model_weight.shape[1:]


class GuestDenseModel(DenseModel):
    def __init__(self):
        super(GuestDenseModel, self).__init__()
        self.role = "guest"

    def forward_dense(self, x):
        LOGGER.debug("interactive guest model weight is {}".format(self.model_weight))
        self.input = x

        output = np.matmul(x, self.model_weight)

        return output

    def get_backward_gradient(self, delta):
        if self.empty:
            return None

        error = np.matmul(delta, self.model_weight.T)

        return error

    def get_gradient(self, delta):
        delta_w = np.matmul(delta.T, self.input)

        return delta_w

    def apply_update(self, delta):
        if self.empty:
            return None

        self.model_weight -= self.lr * delta.T


class HostDenseModel(DenseModel):
    def __init__(self):
        super(HostDenseModel, self).__init__()
        self.role = "host"

    def forward_dense(self, x):
        LOGGER.debug("interactive host model weight is {}".format(self.model_weight))

        self.input = x
        output = x * self.model_weight

        if self.bias:
            output += self.bias

        return output

    def get_backward_gradient(self, delta, acc_noise):
        error = np.matmul(delta, (self.model_weight.T + acc_noise))

        return error

    def get_gradient(self, delta):
        delta_w = np.matmul(delta.T, self.input.to_numpy_array())

        return delta_w

    def apply_update(self, delta):
        delta_w = np.matmul(delta.T, self.input)
        self.model_weight -= self.lr * delta_w.T

        if self.bias is not None:
            self.bias -= self.lr * np.mean(delta, axis=0)

    def update_weight(self, delta):
        self.model_weight -= delta.T * self.lr

    def update_bias(self, delta):
        self.bias -= delta


class InterActiveGuestDenseLayer(object):

    def __init__(self, params=None, layer_config=None, sess=None):
        self.nn_define = layer_config
        self.layer_config = layer_config.get("config").get("layers")[0]

        self.host_input_shape = None
        self.guest_input_shape = None
        self.model = None
        self.rng_generator = random_number_generator.RandomNumberGenerator(method=params.random_param.method,
                                                                           seed=params.random_param.seed,
                                                                           loc=params.random_param.loc,
                                                                           scale=params.random_param.scale)
        self.transfer_variable = None
        self.learning_rate = params.interactive_layer_lr
        self.encrypted_host_dense_output = None
        self.sess = sess

        self.encrypted_host_input = None
        self.guest_input = None
        self.guest_output = None
        self.host_output = None

        self.dense_output_data = None

        self.guest_model = None
        self.host_model = None

        self.partitions = 0

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_partition(self, partition):
        self.partitions = partition

    def __build_model(self, host_input_shape, guest_input_shape):
        self.host_model = HostDenseModel()
        self.host_model.build(host_input_shape, self.layer_config, self.sess)

        self.guest_model = GuestDenseModel()
        self.guest_model.build(guest_input_shape, self.layer_config, self.sess)

    def forward(self, guest_input, iters=0, batch=0):
        encrypted_host_input = HeteroNNTensor(tb_obj=self.get_host_encrypted_forward_from_host(iters, batch))

        if not self.partitions:
            self.partitions = encrypted_host_input.partitions

        self.encrypted_host_input = encrypted_host_input
        self.guest_input = guest_input

        if self.guest_model is None:
            host_input_shape = encrypted_host_input.shape[1]
            guest_input_shape = guest_input.shape[1] if guest_input is not None else 0
            self.__build_model(host_input_shape, guest_input_shape)

        host_output = self.forward_interactive(encrypted_host_input, iters, batch)

        LOGGER.debug("guest input' shape is {}".format(guest_input.shape))
        guest_output = self.guest_model.forward_dense(guest_input)
        LOGGER.debug("guest output' shape is {}".format(guest_output.shape))

        if not self.guest_model.empty:
            dense_output_data = host_output + HeteroNNTensor(ori_data=guest_output, partitions=self.partitions)
        else:
            dense_output_data = host_output

        self.dense_output_data = dense_output_data

        LOGGER.debug("dense output shape is {}".format(self.dense_output_data.shape))
        self.guest_output = guest_output
        self.host_output = host_output

        LOGGER.debug("dense output data ndarray' shape is {}".format(self.dense_output_data.to_numpy_array().shape))
        activation_out = self.host_model.forward_activation(self.dense_output_data.to_numpy_array())
        LOGGER.debug("activation output is {}".format(activation_out.shape))

        return activation_out

    def backward(self, gradient, iters, batch):
        LOGGER.debug("backward_gradient[0]'s shape {}".format(gradient[0].shape))
        activation_backward = self.host_model.backward_activation()[0]
        LOGGER.debug("activation_backward'shape {}".format(activation_backward.shape))
        # gradient = HeteroNNTensor(ori_data=gradient[0], partitions=self.partitions)
        gradient = gradient[0]

        gradient *= activation_backward
        LOGGER.debug("backward_gradient's shape {}".format(gradient.shape))
        LOGGER.debug("backward_gradient's value {}".format(gradient))
        # gradient = gradient.multiply(HeteroNNTensor(ori_data=activation_backward, partitions=self.partitions))

        guest_error = self.update_guest(gradient)

        weight_gradient, acc_noise = self.backward_interactive(gradient, iters, batch)

        host_error = self.update_host(gradient, weight_gradient, acc_noise)

        self.send_host_backward_to_host(host_error, iters, batch)

        return guest_error

    def send_host_backward_to_host(self, host_error, iters, batch):
        self.transfer_variable.host_backward.remote(host_error,
                                                    role=consts.HOST,
                                                    idx=0,
                                                    suffix=(iters, batch,))

    def update_guest(self, gradient):
        back_gradient = self.guest_model.get_backward_gradient(gradient)
        weight_gradient = self.guest_model.get_gradient(gradient)
        self.guest_model.apply_update(weight_gradient)

        return back_gradient

    def update_host(self, gradient, weight_gradient, acc_noise):
        back_gradient = self.host_model.get_backward_gradient(gradient, acc_noise)

        self.host_model.update_weight(weight_gradient)

        return back_gradient

    def forward_interactive(self, encrypted_host_input, iters, batch):
        LOGGER.debug("encrypted_host_input'shape is {}".format(encrypted_host_input.shape))
        LOGGER.debug("model_weight'shape is {}".format(self.host_model.get_weight().shape))
        encrypted_dense_output = self.host_model.forward_dense(encrypted_host_input)

        LOGGER.debug("encrypted_dense_output'shape is {}".format(encrypted_dense_output.shape))
        self.encrypted_host_dense_output = encrypted_dense_output

        guest_forward_noise = self.rng_generator.fast_generate_random_number(encrypted_dense_output.shape,
                                                                             encrypted_dense_output.partitions)
        LOGGER.debug("guest_forward_noise'shape is {}".format(guest_forward_noise.shape))
        # guest_forward_noise = self.rng_generator.generate_random_number(encrypted_dense_output.shape)

        encrypted_dense_output += guest_forward_noise
        LOGGER.debug("encrypted_dense_output'shape is {}".format(encrypted_dense_output.shape))

        self.send_guest_encrypted_forward_output_with_noise_to_host(encrypted_dense_output.get_obj(), iters, batch)

        decrypted_dense_output = self.get_guest_decrypted_forward_from_host(iters, batch)

        return HeteroNNTensor(tb_obj=decrypted_dense_output) - guest_forward_noise

    def backward_interactive(self, delta, iters, batch):
        encrypted_delta_w = self.host_model.get_gradient(delta)
        # encrypted_delta_w = np.matmul(delta.T, self.encrypted_host_input)

        noise_w = self.rng_generator.generate_random_number(encrypted_delta_w.shape)
        self.transfer_variable.encrypted_guest_weight_gradient.remote(encrypted_delta_w + noise_w,
                                                                      role=consts.HOST,
                                                                      idx=-1,
                                                                      suffix=(iters, batch,))

        decrypted_delta_w = self.transfer_variable.decrypted_guest_weight_gradient.get(idx=0,
                                                                                       suffix=(iters, batch,))

        decrypted_delta_w -= noise_w

        encrypted_acc_noise = self.get_encrypted_acc_noise_from_host(iters, batch)

        return decrypted_delta_w, encrypted_acc_noise

    def get_host_encrypted_forward_from_host(self, iters, batch):
        return self.transfer_variable.encrypted_host_forward.get(idx=0,
                                                                 suffix=(iters, batch,))

    def send_guest_encrypted_forward_output_with_noise_to_host(self, encrypted_guest_forward_with_noise, iters, batch):
        return self.transfer_variable.encrypted_guest_forward.remote(encrypted_guest_forward_with_noise,
                                                                     role=consts.HOST,
                                                                     idx=-1,
                                                                     suffix=(iters, batch,))

    def get_guest_decrypted_forward_from_host(self, iters, batch):
        return self.transfer_variable.decrypted_guest_fowrad.get(idx=0,
                                                                 suffix=(iters, batch,))

    def get_encrypted_acc_noise_from_host(self, iters, batch):
        return self.transfer_variable.encrypted_acc_noise.get(idx=0,
                                                              suffix=(iters, batch,))

    def get_output_shape(self):
        return self.host_model.output_shape

    def export_model(self, model_builder):
        input_shape = self.host_model.get_input_shape()
        input_shape += self.guest_model.get_input_shape()
        keras_model = model_builder(input_shape=input_shape,
                                    nn_define=self.nn_define,
                                    optimizer=SimpleNamespace(optimizer="SGD", kwargs={}),
                                    loss="mse",
                                    metrics=None,
                                    sess=self.sess)

        weights = np.concatenate((self.host_model.get_weight(), self.guest_model.get_weight()), axis=0)

        layer_weights = [weights]
        if self.host_model.get_bias() is not None:
            layer_weights.append(self.host_model.get_bias())

        keras_model._model.layers[0].set_weights(layer_weights)

        return keras_model.export_model()


class InteractiveHostDenseLayer(object):
    def __init__(self, param):
        self.acc_noise = None
        self.learning_rate = param.interactive_layer_lr
        self.encrypter = self.generate_encrypter(param)
        self.transfer_variable = None
        self.partitions = 1
        self.rng_generator = random_number_generator.RandomNumberGenerator(method=param.random_param.method,
                                                                           seed=param.random_param.seed,
                                                                           loc=param.random_param.loc,
                                                                           scale=param.random_param.scale)

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_partition(self, partition):
        self.partitions = partition

    def forward(self, host_input, iters=0, batch=0):
        host_input = HeteroNNTensor(ori_data=host_input, partitions=self.partitions)
        LOGGER.debug("encrypted_host_input'shape is {}".format(host_input.shape))
        encrypted_host_input = host_input.encrypt(self.encrypter)
        LOGGER.debug("encrypted_host_input'shape is {}".format(encrypted_host_input.shape))
        # encrypted_host_input = self.encrypter.recursive_encrypt(host_input)
        self.send_host_encrypted_forward_to_guest(encrypted_host_input.get_obj(), iters, batch)

        encrypted_guest_forward = HeteroNNTensor(tb_obj=self.get_guest_encrypted_forwrad_from_guest(iters, batch))

        LOGGER.debug("encrypted_guest_forward's shape is {}".format(encrypted_guest_forward.shape))
        decrypted_guest_forward = encrypted_guest_forward.decrypt(self.encrypter)
        LOGGER.debug("decrypted_guest_forward's shape is {}".format(decrypted_guest_forward.shape))
        # decrypted_guest_forward = self.encrypter.recursive_decrypt(encrypted_guest_forward)

        """TODO: optimize acc_noise to vector"""
        if self.acc_noise is None:
            # self.acc_noise = np.zeros(decrypted_guest_forward.shape)
            self.acc_noise = 0

        decrypted_guest_forward_with_noise = decrypted_guest_forward + self.acc_noise * self.learning_rate

        self.send_decrypted_guest_forward_with_noise_to_guest(decrypted_guest_forward_with_noise.get_obj(), iters,
                                                              batch)

    def backward(self, iters, batch):
        encrypted_guest_weight_gradient = self.get_guest_encrypted_weight_gradient_from_guest(iters, batch)

        decrypted_guest_weight_gradient = self.encrypter.recursive_decrypt(encrypted_guest_weight_gradient)

        """TODO: optimize acc_noise to vector"""
        # noise_weight_gradient = self.rng_generator.generate_random_number(decrypted_guest_weight_gradient.shape)
        noise_weight_gradient = self.rng_generator.generate_random_number((1,))[0]

        decrypted_guest_weight_gradient += self.learning_rate * noise_weight_gradient / self.learning_rate

        self.send_guest_decrypted_weight_gradient_to_guest(decrypted_guest_weight_gradient, iters, batch)

        encrypted_acc_noise = self.encrypter.encrypt(self.acc_noise)
        self.send_encrypted_acc_noise_to_guest(encrypted_acc_noise, iters, batch)

        self.acc_noise += noise_weight_gradient
        host_backward = self.get_host_backward_from_guest(iters, batch)

        host_backward = self.encrypter.recursive_decrypt(host_backward)
        return host_backward

    def send_encrypted_acc_noise_to_guest(self, encrypted_acc_noise, iters, batch):
        self.transfer_variable.encrypted_acc_noise.remote(encrypted_acc_noise,
                                                          idx=0,
                                                          role=consts.GUEST,
                                                          suffix=(iters, batch,))

    def get_guest_encrypted_weight_gradient_from_guest(self, iters, batch):
        encrypted_guest_weight_gradient = self.transfer_variable.encrypted_guest_weight_gradient.get(idx=0,
                                                                                                     suffix=(
                                                                                                         iters, batch,))

        return encrypted_guest_weight_gradient

    def send_host_encrypted_forward_to_guest(self, encrypted_host_input, iters, batch):
        self.transfer_variable.encrypted_host_forward.remote(encrypted_host_input,
                                                             idx=0,
                                                             role=consts.GUEST,
                                                             suffix=(iters, batch,))

    def send_guest_decrypted_weight_gradient_to_guest(self, decrypted_guest_weight_gradient, iters, batch):
        self.transfer_variable.decrypted_guest_weight_gradient.remote(decrypted_guest_weight_gradient,
                                                                      idx=0,
                                                                      role=consts.GUEST,
                                                                      suffix=(iters, batch,))

    def get_host_backward_from_guest(self, iters, batch):
        host_backward = self.transfer_variable.host_backward.get(idx=0,
                                                                 suffix=(iters, batch,))

        return host_backward

    def get_guest_encrypted_forwrad_from_guest(self, iters, batch):
        encrypted_guest_forward = self.transfer_variable.encrypted_guest_forward.get(idx=0,
                                                                                     suffix=(iters, batch,))

        return encrypted_guest_forward

    def send_decrypted_guest_forward_with_noise_to_guest(self, decrypted_guest_forward_with_noise, iters, batch):
        self.transfer_variable.decrypted_guest_fowrad.remote(decrypted_guest_forward_with_noise,
                                                             idx=0,
                                                             role=consts.GUEST,
                                                             suffix=(iters, batch,))

    def generate_encrypter(self, param):
        LOGGER.info("generate encrypter")
        if param.encrypt_param.method.lower() == consts.PAILLIER.lower():
            encrypter = PaillierEncrypt()
            encrypter.generate_key(param.encrypt_param.key_length)
        else:
            raise NotImplementedError("encrypt method not supported yet!!!")

        return encrypter

    def export_model(self):
        return None
