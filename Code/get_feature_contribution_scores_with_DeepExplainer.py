#!/usr/bin/env python3

from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import json
import h5py
from basenji import seqnn_rna
import shap
import argparse
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", "-d", help="dir of model inputs", required=True)
    parser.add_argument("--params", "-p", help="params file", required=True)
    parser.add_argument("--model", "-m", help="model file", required=True)
    parser.add_argument("--label", "-l", help="dataset to get contribution scores", required=True)
    parser.add_argument("--regionstart", "-r", type=int, help="start index of train data to get feature contributions")
    parser.add_argument("--regionend", "-e", type=int, help="start index of train data to get feature contributions")
    parser.add_argument("--numprocess", "-n", type=int, help="mumber of parallel processes", required=True)
    parser.add_argument( "--outdir","-o",help="dir to write contribution scores",required=True)
    args = parser.parse_args()

    num_processes = int(args.numprocess)
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

    with open(args.params) as params_open:
        params = json.load(params_open)
    params_model = params['model']
    
    seqnn_model_pre = seqnn_rna.SeqNN(params_model)
    seqnn_model_pre.restore(args.model,0)
    seqnn_model = seqnn_model_pre.model

    train_seq_inputs = h5py.File(f'{args.datadir}/train_seq_inputs.h5','r')['seq_inputs'][:,:,:]
    train_rna_inputs = h5py.File(f'{args.datadir}/train_rna_inputs.h5','r')['rna_inputs'][:,:,:]
    if(args.label != "train"):
        fout = h5py.File(f'{args.outdir}/{args.label}_scores.h5','w')
        seq_inputs = h5py.File(f'{args.datadir}/{args.label}_seq_inputs.h5','r')['seq_inputs'][:,:,:]
        rna_inputs = h5py.File(f'{args.datadir}/{args.label}_rna_inputs.h5','r')['rna_inputs'][:,:,:]
        background_index = np.random.choice(train_seq_inputs.shape[0], 20, replace=False) 
        background = train_seq_inputs[background_index]
        background_rna = train_rna_inputs[background_index]
    else:
        fout = h5py.File(f'{args.outdir}/train_scores_{args.regionstart}_{args.regionend}.h5','w')
        half = 3500
        if((args.regionstart) and (args.regionend)):
            seq_inputs = train_seq_inputs[args.regionstart:args.regionend,:,:]
            rna_inputs = train_rna_inputs[args.regionstart:args.regionend,:,:] 
            if(region_end<=3500):
                background_seq_inputs = train_seq_inputs[half:,:,:]
                background_rna_inputs = train_rna_inputs[half:,:,:]
            else:
                background_seq_inputs = train_seq_inputs[0:half,:,:]
                background_rna_inputs = train_rna_inputs[0:half,:,:]
            background_index = np.random.choice(background_seq_inputs.shape[0], 20, replace=False)
            background = background_seq_inputs[background_index]
            background_rna = background_rna_inputs[background_index]
        else:
            parser.error('Must provide start and end indexes of the subset of train data')
        
    dinuc_shuff_explainer = shap.DeepExplainer(seqnn_model, [background,background_rna])
    raw_shap_explanations = dinuc_shuff_explainer.shap_values([seqs_inputs,rna_inputs])
    num_seqs = seq_inputs.shape[0]
    shap_explanations_seq = np.zeros((num_seqs, seq_inputs.shape[1], 4), dtype='float16')
    shap_explanations_hyp_seq = raw_shap_explanations[0]
    shap_explanations_rna = raw_shap_explanations[1]
    for i in range(num_seqs):
        shap_explanations_seq[i,:,:] = shap_explanations_hyp_seq[i,:,:]*seq_inputs[i]

    fout.create_dataset("seq_contrib_scores", data=shap_explanations_seq)
    fout.create_dataset("seq_hyp_contrib_scores", data=shap_explanations_hyp_seq)
    fout.create_dataset("rna_contrib_scores", data=shap_explanations_rna)
    fout.close()

if __name__ == '__main__':
    main()
