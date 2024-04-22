#!/usr/bin/env python
# Modified based on akita_data_write.py

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser
import os
import sys

import h5py
import pandas as pd
import numpy as np
import pdb
import pysam

from basenji_data import ModelSeq
from basenji.dna_io import dna_1hot

import tensorflow as tf

"""
akitaR_data_write.py

Write TF Records for batches of model sequences and rna/epi features.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <seqs_bed_file> <seqs_cov_dir> <rna_file> <tfr_file>'
  parser = OptionParser(usage)
  parser.add_option('-g', dest='genome_index',
      default=None, type='int', help='Genome index')
  parser.add_option('-s', dest='start_i',
      default=0, type='int',
      help='Sequence start index [Default: %default]')
  parser.add_option('-e', dest='end_i',
      default=None, type='int',
      help='Sequence end index [Default: %default]')
  parser.add_option('--te', dest='target_extend',
      default=None, type='int', help='Extend targets vector [Default: %default]')
  parser.add_option('--ts', dest='target_start',
      default=0, type='int', help='Write targets into vector starting at index [Default: %default')
  parser.add_option('-w',dest='pool_width',
      default=1, type='int',
      help='Average pooling width [Default: %default]')
  parser.add_option('-n', dest='number_features',
      default=1, type='int',
      help='Number of rna/epi features included in the RNA file [Default: %default]')
  parser.add_option('-u', dest='umap_npy',
      help='Unmappable array numpy file')
  parser.add_option('--umap_set', dest='umap_set',
      default=None, type='float',
      help='Sequence distribution value to set unmappable positions to, eg 0.25.')
  (options, args) = parser.parse_args()

  if len(args) != 5:
    parser.error('Must provide input arguments.')
  else:
    fasta_file = args[0]
    seqs_bed_file = args[1]
    seqs_cov_dir = args[2]
    rna_file = args[3]
    tfr_file = args[4]

  ################################################################
  # read model sequences

  model_seqs = []
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]),None))

  if options.end_i is None:
    options.end_i = len(model_seqs)

  num_seqs = options.end_i - options.start_i

  ################################################################
  # determine sequence coverage files

  seqs_cov_files = []
  ti = 0
  if options.genome_index is None:
    seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)
  else:
    seqs_cov_file = '%s/%d-%d.h5' % (seqs_cov_dir, options.genome_index, ti)
  while os.path.isfile(seqs_cov_file):
    seqs_cov_files.append(seqs_cov_file)
    ti += 1
    if options.genome_index is None:
      seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)
    else:
      seqs_cov_file = '%s/%d-%d.h5' % (seqs_cov_dir, options.genome_index, ti)

  if len(seqs_cov_files) == 0:
    print('Sequence coverage files not found, e.g. %s' % seqs_cov_file, file=sys.stderr)
    exit(1)

  seq_pool_len_hic = h5py.File(seqs_cov_files[0], 'r')['targets'].shape[1]
  num_targets = len(seqs_cov_files)
  rna_df = pd.read_csv(rna_file, index_col=0, sep='\t')
  num_features = options.number_features
  ################################################################
  # read targets

  # extend targets
  num_targets_tfr = num_targets
  if options.target_extend is not None:
    assert(options.target_extend >= num_targets_tfr)
    num_targets_tfr = options.target_extend

  # initialize targets
  targets = np.zeros((num_seqs, seq_pool_len_hic, num_targets_tfr), dtype='float16')
  
  seq_len_nt = model_seqs[0].end - model_seqs[0].start
  seq_len_pool = seq_len_nt // options.pool_width
  rnas = np.zeros((num_seqs, seq_len_pool, num_features), dtype='float16')

  # read each target
  for ti in range(num_targets):
    seqs_cov_open = h5py.File(seqs_cov_files[ti], 'r')
    tii = options.target_start + ti
    targets[:,:,tii] = seqs_cov_open['targets'][options.start_i:options.end_i,:]
    seqs_cov_open.close()

  col_index_s = 0
  col_index_e = 0
  for ti in range(rna_df.shape[0]):    
    rna_level = pd.read_table(rna_df['file'].iloc[ti],sep="\t",header=0,index_col=None)
    col_index_e += (rna_level.shape[1]-3)
    for si in range(num_seqs):
        msi = options.start_i + si
        mseq = model_seqs[msi]
        rl = rna_level.index[(rna_level['chrom']==mseq.chr) & (rna_level['start']==mseq.start)][0]
        rh = rna_level.index[(rna_level['chrom']==mseq.chr) & (rna_level['start']==mseq.end)][0]
        if(rna_df['description'].iloc[ti]=="TransRNA"):
          rnas[si,:,col_index_s:col_index_e] = np.log(rna_level.iloc[rl:rh,3:]/float(rna_df['norm'].iloc[ti])+1)
        else:
          rnas[si,:,col_index_s] = np.log(rna_level['sum'][rl:rh]/float(rna_df['norm'].iloc[ti])+1)

    col_index_s = col_index_e


  ################################################################
  # write TFRecords

  # open FASTA
  fasta_open = pysam.Fastafile(fasta_file)

  # define options
  tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')

  with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
    for si in range(num_seqs):
      msi = options.start_i + si
      mseq = model_seqs[msi]

      # read FASTA
      seq_dna = fasta_open.fetch(mseq.chr, mseq.start, mseq.end)

      # one hot code
      seq_1hot = dna_1hot(seq_dna)

      if options.genome_index is None:
        example = tf.train.Example(features=tf.train.Features(feature={
            'genome': _int_feature(0),
            'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
            'rna': _bytes_feature(rnas[si,:,:].flatten().tostring()),
            'target': _bytes_feature(targets[si,:,:].flatten().tostring())}))
      else:
        example = tf.train.Example(features=tf.train.Features(feature={
            'genome': _int_feature(options.genome_index),
            'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
            'rna': _bytes_feature(rnas[si,:,:].flatten().tostring()),
            'target': _bytes_feature(targets[si,:,:].flatten().tostring())}))

      writer.write(example.SerializeToString())

    fasta_open.close()


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
