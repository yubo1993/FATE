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

from federatedml.nn.homo_nn.backend.tf_keras.layers import get_builder
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.backend import gradients
from federatedml.util import consts
from federatedml.nn.hetero_nn.losses import keep_predict_loss
from federatedml.nn.hetero_nn.util import random_number_generator
from federatedml.transfer_variable.transfer_class.hetero_nn_transfer_variable import HeteroNNTransferVariable
from federatedml.secureprotol import PaillierEncrypt
from arch.api.utils import log_utils
import numpy as np
import tensorflow as tf

LOGGER = log_utils.getLogger()


"""
TODO:
      1. make Guest\Host a signle object, to support guest' input_shape = 0
      2. Parallelize
      3. change Acc_noise to vector 
"""
class InterActiveGuestDenseLayer(object):
    def __init__(self, params=None, layer_config=None, sess=None):
        self.layer_config = layer_config.get("config").get("layers")[0]

        self.host_input_shape = None
        self.guest_input_shape = None
        self.model = None
        self.rng_generator = random_number_generator.RandomNumberGenerator(method=params.random_param.method,
                                                                           seed=params.random_param.seed,
                                                                           loc=params.random_param.loc,
                                                                           scale=params.random_param.scale)
        self.transfer_variable = None
        self.activation_func = None
        self.activation_placeholder_name = "activation_placeholder"
        self.activation_gradient_func = None
        self.learning_rate = params.interactive_layer_lr
        self.encrypted_host_dense_output = None
        self.sess = sess

        self.encrypted_host_input = None
        self.guest_input = None
        self.guest_output = None
        self.host_output = None

        self.dense_output_data = None
        self.guest_model_weight = None
        self.host_model_weight = None
        self.bias = None

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def __build_model(self, input_shape, role="guest", mode="weight"):
        model = Sequential()

        if "layer" in self.layer_config:
            del self.layer_config["layer"]
        self.layer_config["input_shape"] = input_shape
        builder = get_builder("Dense")
        model.add(builder(**self.layer_config["config"]))
        model.compile("SGD", loss=keep_predict_loss)

        self.sess.run(tf.initialize_all_variables())
        if mode == "weight":
            trainable_weights = self.sess.run(model.trainable_weights)
            setattr(self, "_".join([role, "model_weight"]), trainable_weights[0])

            if role == "host":
                if model.layers[0].use_bias:
                    self.bias = trainable_weights[1]
        else:
            self.activation_func = model.layers[0].activation
            self.__build_activation_layer_gradients_func(model)

    def forward(self, guest_input, iters=0, batch=0):
        encrypted_host_input = self.get_host_encrypted_forward_from_host(iters, batch)

        self.encrypted_host_input = encrypted_host_input
        self.guest_input = guest_input

        if self.guest_model_weight is None:
            self.__build_model(encrypted_host_input.shape[0], role="host")
            self.__build_model(guest_input.shape[0], role="guest")
            self.__build_model(self.guest_model_weight.shape[1], mode="activation")

        host_output = self.forward_interactive(encrypted_host_input, iters, batch)

        guest_output = np.matmul(guest_input, self.guest_model_weight)
        dense_output_data = host_output + guest_output

        if self.bias:
            dense_output_data += self.bias

        self.dense_output_data = dense_output_data
        self.guest_output = guest_output
        self.host_output = host_output

        return self.activation_func(dense_output_data)

    def backward(self, gradient, iters, batch):
        gradient = gradient[0]
        activation_backward = self.calculate_activation_backward(self.dense_output_data)[0]

        gradient *= activation_backward

        guest_error = self.update_guest(gradient)

        weight_gradient = self.backward_interactive(gradient, iters, batch)

        host_error = self.update_host(gradient, weight_gradient)

        self.send_host_backward_to_host(host_error, iters, batch)

        return guest_error

    def send_host_backward_to_host(self, host_error, iters, batch):
        self.transfer_variable.host_backward.remote(host_error,
                                                    role=consts.HOST,
                                                    idx=0,
                                                    suffix=(iters, batch,))

    def update_guest(self, gradient):
        delta_w = np.matmul(gradient.T, self.guest_input)

        error = np.matmul(gradient, self.guest_model_weight.T)
        self.guest_model_weight -= self.learning_rate * delta_w.T

        return error

    def update_host(self, gradient, weight_gradient):
        # host_error = np.matmul(self.encrypted_host_input, self.host_model_weight)
        host_error = np.matmul(gradient, self.host_model_weight.T)
        self.host_model_weight -= weight_gradient.T * self.learning_rate

        return host_error

    def forward_interactive(self, encrypted_host_input, iters, batch):
        encrypted_dense_output = np.matmul(encrypted_host_input, self.host_model_weight)
        self.encrypted_host_dense_output = encrypted_dense_output

        guest_forward_noise = self.rng_generator.generate_random_number(encrypted_dense_output.shape)

        encrypted_dense_output += guest_forward_noise

        self.send_guest_encrypted_forward_output_with_noise_to_host(encrypted_dense_output, iters, batch)

        decrypted_dense_output = self.get_guest_decrypted_forward_from_host(iters, batch)

        return decrypted_dense_output

    def backward_interactive(self, delta, iters, batch):
        # noise_delta = self.rng_generator.generate_random_number(delta.shape)
        encrypted_delta_w = np.matmul(delta.T, self.encrypted_host_input)
        noise_w = self.rng_generator.generate_random_number(encrypted_delta_w.shape)
        self.transfer_variable.encrypted_guest_weight_gradient.remote(encrypted_delta_w + noise_w,
                                                                      role=consts.HOST,
                                                                      idx=-1,
                                                                      suffix=(iters, batch,))

        decrypted_delta_w = self.transfer_variable.decrypted_guest_weight_gradient.get(idx=0,
                                                                                       suffix=(iters, batch,))

        decrypted_delta_w -= noise_w

        return decrypted_delta_w

    def __build_activation_layer_gradients_func(self, model):
        dense_layer = model.layers[0]
        shape = dense_layer.output_shape
        dtype = dense_layer.get_weights()[0].dtype

        input_data = tf.placeholder(shape=shape,
                                    dtype=dtype,
                                    name=self.activation_placeholder_name)

        self.activation_gradient_func = gradients(dense_layer.activation(input_data), input_data)

    def calculate_activation_backward(self, dense_output):
        placeholder = tf.get_default_graph().get_tensor_by_name(":".join([self.activation_placeholder_name, "0"]))
        return self.sess.run(self.activation_gradient_func,
                             feed_dict={placeholder: dense_output})

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

    def get_output_shape(self):
        return self.guest_model_weight.shape[1:]

    def export_model(self, model_builder):
        input_shape = self.host_input_shape
        input_shape[0] += self.guest_input_shape
        keras_model = model_builder(input_shape=input_shape,
                                    nn_define=self.layer_config,
                                    optimizer="SGD",
                                    loss="mse",
                                    metrics=None,
                                    sess=self.sess)

        weights = np.concatenate((self.host_model_weight, self.guest_model_weight), axis=1)
        keras_model._model.layers[0].set_weights(weights)

        return keras_model.export_model()


class InteractiveHostDenseLayer(object):
    def __init__(self, param):
        self.acc_noise = None
        self.learning_rate = param.interactive_layer_lr
        self.encrypter = self.generate_encrypter(param)
        self.transfer_variable = None
        self.rng_generator = random_number_generator.RandomNumberGenerator(method=param.random_param.method,
                                                                           seed=param.random_param.seed,
                                                                           loc=param.random_param.loc,
                                                                           scale=param.random_param.scale)

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def forward(self, host_input, iters=0, batch=0):
        encrypted_host_input = self.encrypter.recursive_encrypt(host_input)
        self.send_host_encrypted_forward_to_guest(encrypted_host_input, iters, batch)

        encrypted_guest_forward = self.get_guest_encrypted_forwrad_from_guest(iters, batch)

        decrypted_guest_forward = self.encrypter.recursive_decrypt(encrypted_guest_forward)

        """TODO: optimize acc_noise to vector"""
        if self.acc_noise is None:
            # self.acc_noise = np.zeros(decrypted_guest_forward.shape)
            self.acc_noise = 0

        decrypted_guest_forward_with_noise = decrypted_guest_forward + self.acc_noise * self.learning_rate

        self.send_decrypted_guest_forward_with_noise_to_guest(decrypted_guest_forward_with_noise, iters, batch)

    def backward(self, iters, batch):
        encrypted_guest_weight_gradient = self.get_guest_encrypted_weight_gradient_from_guest(iters, batch)

        decrypted_guest_weight_gradient = self.encrypter.recursive_decrypt(encrypted_guest_weight_gradient)

        """TODO: optimize acc_noise to vector"""
        # noise_weight_gradient = self.rng_generator.generate_random_number(decrypted_guest_weight_gradient.shape)
        noise_weight_gradient = self.rng_generator.generate_random_number((1, ))[0]

        self.acc_noise += noise_weight_gradient

        decrypted_guest_weight_gradient += self.learning_rate * noise_weight_gradient

        self.send_guest_decrypted_weight_gradient_to_guest(decrypted_guest_weight_gradient, iters, batch)

        host_backward = self.get_host_backward_from_guest(iters, batch)

        return host_backward

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
            raise NotImplementedError("encrypt method not supported yes!!!")

        return encrypter
