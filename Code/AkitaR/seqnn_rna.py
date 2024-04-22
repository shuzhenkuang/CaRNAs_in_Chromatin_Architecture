# Modified based on seqnn.py of basenji
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
from __future__ import print_function

import pdb
import sys
import time

from natsort import natsorted
import numpy as np
import tensorflow as tf

import blocks_rna as blocks
import layers_rna as layers
from basenji import metrics
from basenji.seqnn import SeqNN


class SeqRNANN(SeqNN):
  def __init__(self, params):
    super().__init__(params)


  def build_model(self, save_reprs=False):
    ###################################################
    # inputs
    ###################################################
    sequence = tf.keras.Input(shape=(self.seq_length, 4), name='seq_input')
    current = sequence

    rna = tf.keras.Input(shape=(self.target_length, self.feature_length), name='rna_input')
    rna_current = rna
    # augmentation
    if self.augment_rc:
      current, rna_current, reverse_bool = layers.StochasticReverseComplement()(current,rna_current)
    if self.augment_shift != [0]:
      current = layers.StochasticShift(self.augment_shift)(current)
    self.preds_triu = False
    
    ###################################################
    # build convolution blocks
    ###################################################
    for bi, block_params in enumerate(self.trunk):
      if(block_params['name'] == 'dilated_residual'):
        current = tf.keras.layers.Concatenate()([current,rna_current])
      current = self.build_block(current, block_params)

    # final activation
    current = layers.activate(current, self.activation)

    # make model trunk
    trunk_output = current
    self.model_trunk = tf.keras.Model(inputs=[sequence, rna], outputs=trunk_output)

    ###################################################
    # heads
    ###################################################
    head_keys = natsorted([v for v in vars(self) if v.startswith('head')])
    self.heads = [getattr(self, hk) for hk in head_keys]

    self.head_output = []
    for hi, head in enumerate(self.heads):
      if not isinstance(head, list):
        head = [head]

      # reset to trunk output
      current = trunk_output

      # build blocks
      for bi, block_params in enumerate(head):
        current = self.build_block(current, block_params)

      # transform back from reverse complement
      if self.augment_rc:
        if self.preds_triu:
          current = layers.SwitchReverseTriu(self.diagonal_offset)([current, reverse_bool])
        else:
          current = layers.SwitchReverse()([current, reverse_bool])

      # save head output
      self.head_output.append(current)

    ###################################################
    # compile model(s)
    ###################################################
    self.models = []
    for ho in self.head_output:
      self.models.append(tf.keras.Model(inputs=[sequence, rna], outputs=ho))
    self.model = self.models[0]
    print(self.model.summary())

    ###################################################
    # track pooling/striding and cropping
    ###################################################
    self.model_strides = []
    self.target_lengths = []
    self.target_crops = []
    for model in self.models:
      self.model_strides.append(1)
      for layer in self.model.layers:
        if hasattr(layer, 'strides'):
          self.model_strides[-1] *= layer.strides[0]
      if type(sequence.shape[1]) == tf.compat.v1.Dimension:
        target_full_length = sequence.shape[1].value // self.model_strides[-1]
      else:
        target_full_length = sequence.shape[1] // self.model_strides[-1]

      self.target_lengths.append(model.outputs[0].shape[1])
      if type(self.target_lengths[-1]) == tf.compat.v1.Dimension:
        self.target_lengths[-1] = self.target_lengths[-1].value
      self.target_crops.append((target_full_length - self.target_lengths[-1])//2)
    print('model_strides', self.model_strides)
    print('target_lengths', self.target_lengths)
    print('target_crops', self.target_crops)


  def build_ensemble(self, ensemble_rc=False, ensemble_shifts=[0]):
    """ Build ensemble of models computing on augmented input sequences. """
    if ensemble_rc or len(ensemble_shifts) > 1:
      # sequence input
      sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')
      sequences = [sequence]
      
      rna = tf.keras.Input(shape=(self.target_length, self.feature_length), name='rna_input')
      rnas = [rna]

      if len(ensemble_shifts) > 1:
        # generate shifted sequences
        sequences = layers.EnsembleShift(ensemble_shifts)(sequences)

      if ensemble_rc:
        # generate reverse complements and indicators
        sequences_rev, rnas_rev = layers.EnsembleReverseComplement()(sequences)
      else:
        sequences_rev = [(seq,tf.constant(False)) for seq in sequences]
        rnas_rev = [(rna,tf.constant(False)) for rna in rnas] 

      # predict each sequence
      if self.preds_triu:
        preds = [layers.SwitchReverseTriu(self.diagonal_offset)
                  ([self.model([seq[0],r[0]]), seq[1]]) for (seq,r) in zip(sequences_rev,rnas_rev)]
      else:
        preds = [layers.SwitchReverse()([self.model([seq[0],r[0]]), seq[1]]) for (seq,r) in zip(sequences_rev,rnas_rev)]

      # create layer
      preds_avg = tf.keras.layers.Average()(preds)

      # create meta model
      self.ensemble = tf.keras.Model(inputs=[sequence, rna], outputs=preds_avg)


  def save(self, model_file, trunk=False):
    if trunk:
      self.model_trunk.save(model_file, include_optimizer=False)
    else:
      self.model.save(model_file, include_optimizer=False)
