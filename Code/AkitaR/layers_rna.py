# Modifed based on layers.py of basenji
# Copyright 2019 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import sys
from typing import Optional, List
from basenji.layers import *

import numpy as np
import tensorflow as tf

############################################################
# Augmentation
############################################################

class EnsembleReverseComplement(tf.keras.layers.Layer):
  """Expand tensor to include reverse complement of one hot encoded DNA sequence."""
  def __init__(self):
    super(EnsembleReverseComplement, self).__init__()

  def call(self, seqs_1hot, rnas):
    if not isinstance(seqs_1hot, list):
      seqs_1hot = [seqs_1hot]

    if not isinstance(rnas, list):
      rnas = [rnas]
    
    ens_seqs_1hot = []
    for seq_1hot in seqs_1hot:
      rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
      rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
      ens_seqs_1hot += [(seq_1hot, tf.constant(False)), (rc_seq_1hot, tf.constant(True))]

    ens_rnas = []    
    for rna in rnas:
        rc_rna = tf.reverse(rna, axis=[1])
        ens_rnas += [(rna, tf.constant(False)), (rc_rna, tf.constant(True))]        

    return ens_seqs_1hot, ens_rnas

class StochasticReverseComplement(tf.keras.layers.Layer):
  """Stochastically reverse complement a one hot encoded DNA sequence."""
  def __init__(self):
    super(StochasticReverseComplement, self).__init__()
  def call(self, seq_1hot, rna, training=None):
    if training:
      rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
      rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
      rc_rna = tf.reverse(rna, axis=[1])
      reverse_bool = tf.random.uniform(shape=[]) > 0.5
      src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
      src_rna = tf.cond(reverse_bool, lambda: rc_rna, lambda: rna)
      return src_seq_1hot, src_rna, reverse_bool
    else:
      return seq_1hot, rna, tf.constant(False)


class StochasticShift(tf.keras.layers.Layer):
  """Stochastically shift a one hot encoded DNA sequence."""
  def __init__(self, shift_max=0, symmetric=True, pad='uniform'):
    super(StochasticShift, self).__init__()
    self.shift_max = shift_max
    self.symmetric = symmetric
    if self.symmetric:
      self.augment_shifts = tf.range(-self.shift_max, self.shift_max+1)
    else:
      self.augment_shifts = tf.range(0, self.shift_max+1)
    self.pad = pad

  def call(self, seq_1hot, training=None):
    if training:
      shift_i = tf.random.uniform(shape=[], minval=0, dtype=tf.int64,
                                  maxval=len(self.augment_shifts))
      shift = tf.gather(self.augment_shifts, shift_i)
      sseq_1hot = tf.cond(tf.not_equal(shift, 0),
                          lambda: shift_sequence(seq_1hot, shift),
                          lambda: seq_1hot)
      return sseq_1hot
    else:
      return seq_1hot

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'shift_max': self.shift_max,
      'symmetric': self.symmetric,
      'pad': self.pad
    })
    return config

