# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.utils.types import FloatTensorLike

from typeguard import typechecked
from typing import Union, Callable, Type

def _ref(var):
    return var.ref() if hasattr(var, "ref") else var.experimental_ref()

class KernelDecoupledWeightDecayExtension:
    
    @typechecked
    def __init__(self, weight_decay: Union[FloatTensorLike, Callable], **kwargs):
        
        wd = kwargs.pop("weight_decay", weight_decay)
        super().__init__(**kwargs)
        self._decay_var_list = None
        self._set_hyper("weight_decay", wd)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {"weight_decay": self._serialize_hyperparameter("weight_decay"),}
        )
        return config
    
    def minimize(self, loss, var_list, grad_loss=None, name=None):
        self._decay_var_list = [_ref(i) for i in var_list if 'bias' not in i.name]
        return super().minimize(loss, var_list=var_list, grad_loss=grad_loss, name=name)
    
    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._decay_var_list = [_ref(i[1]) for i in grads_and_vars if 'bias' not in i[1].name]
        return super().apply_gradients(grads_and_vars, name=name, **kwargs)
    
    def _decay_weights_op(self, var):
        if var.ref() in self._decay_var_list:
            return var.assign_sub(
                self._get_hyper("weight_decay", var.dtype) * self._get_hyper("learning_rate", var.dtype) * var, self._use_locking
            )
            # return var - (self._get_hyper("weight_decay", var.dtype) * self._get_hyper("learning_rate") * var)
        return tf.no_op()
    
    def _decay_weights_sparse_op(self, var, indices):
        if not self._decay_var_list or var.ref() in self._decay_var_list:
            update = -self._get_hyper("weight_decay", var.dtype) * self._get_hyper("learning_rate", var.dtype) *tf.gather(
                var, indices
            )
            return self._resource_scatter_add(var, indices, update)
        return tf.no_op()
    
    def _resource_apply_dense(self, grad, var):
        decay_op = self._decay_weights_op(var)
        with tf.control_dependencies([decay_op]):
            return super()._resource_apply_dense(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        decay_op = self._decay_weights_sparse_op(var, indices)
        with tf.control_dependencies([decay_op]):
            return super()._resource_apply_sparse(grad, var, indices)

class SGDW(KernelDecoupledWeightDecayExtension, tf.keras.optimizers.SGD):
    def __init__(self, weight_decay, *args, **kwargs):
        super(SGDW, self).__init__(weight_decay, *args, **kwargs)
        
class AdamW(KernelDecoupledWeightDecayExtension, tf.keras.optimizers.Adam):
    def __init__(self, weight_decay, *args, **kwargs):
        super(AdamW, self).__init__(weight_decay, *args, **kwargs)
