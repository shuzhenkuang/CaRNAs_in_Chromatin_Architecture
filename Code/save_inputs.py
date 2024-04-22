#!/usr/bin/env python3

from __future__ import print_function, division
import os

import h5py
import numpy as np
import pandas as pd
import json
import argparse
from AkitaR import dataset_rna
import tensorflow as tf
if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", "-d", help="Dir of model data",type=str, required=True)
    parser.add_argument("--params", "-p", help="params file", type=str, required=True)
    parser.add_argument("--label", "-l", help="dataset to extract (train/valid/test)", type=str, required=True)
    parser.add_argument("--numprocess", "-n", help="mumber of parallel processes", type=int, required=True)
    parser.add_argument( "--outdir","-o",type=str)
    args = parser.parse_args()
    data_dir = args.datadir
    params_file = args.params
    label = args.label
    num_processes = int(args.numprocess)
    out_dir = args.outdir

    os.environ["OMP_NUM_THREADS"] = str(num_processes)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_processes)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(num_processes)

    tf.config.threading.set_inter_op_parallelism_threads(
        num_processes
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        num_processes
    )
    tf.config.set_soft_device_placement(True)

    with open(f'{data_dir}/{params_file}') as params_open:
        params = json.load(params_open)
    params_train = params['train']

    if(label=="test"):
        split_label = 'test'
        mode = 'eval'
    elif(label=="valid"):
        split_label = 'valid'
        mode = 'eval'
    elif(label=="train"):
        split_label = 'train'
        mode = 'eval'

    eval_data = dataset_rna.SeqDataset(data_dir,
        split_label=split_label,
        batch_size=params_train['batch_size'],
        mode=mode,
        tfr_pattern=None)

    seq_inputs, rna_inputs, targets = eval_data.numpy()

    preds_h5 = h5py.File(f'{out_dir}/{label}_seq_inputs.h5', 'w')
    preds_h5.create_dataset('seq_inputs', data=seq_inputs)
    preds_h5.close()

    preds_h5 = h5py.File(f'{out_dir}/{label}_rna_inputs.h5', 'w')
    preds_h5.create_dataset('rna_inputs', data=rna_inputs)
    preds_h5.close()

    preds_h5 = h5py.File(f'{out_dir}/{label}_targets.h5', 'w')
    preds_h5.create_dataset('targets', data=targets)
    preds_h5.close()

if __name__ == '__main__':
    main()
