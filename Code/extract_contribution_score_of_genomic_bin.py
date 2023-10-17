#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import argparse
import h5py
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()                                               
    parser.add_argument("--shapdir", "-s", help="dir of contribution scores", required=True)
    parser.add_argument("--seqfile", "-f", help="informtion of genomic regions used for train, valid, and test", required=True)
    parser.add_argument( "--regs","-r", nargs='+', help='<Required> Set of selected regions', required=True) 
    parser.add_argument( "--regnames","-n", nargs='+', help='<Required> Set of selected regions', required=True)
    parser.add_argument( "--featurenames","-t", nargs='+', help='<Required> Set of feature names', required=True)
    parser.add_argument( "--outdir","-o", help="dir of output files", required=True)
    args = parser.parse_args()

    if not os.path.isdir(f'{args.outdir}/BinScores'):
        os.makedirs(f'{args.outdir}/BinScores')

    seqs = pd.read_table(args.seqfile,header=0,sep='\t')
    score_files = ['train_rna_scores.h5','valid_rna_scores.h5','test_rna_score.h5'] 
    for i, sf in enumerate(score_files):
        if(i==0):
            rna_scores = h5py.File(f'{args.shap_dir}/{sf}','r')['rna_contrib_scores'][:,:,:]
        else:
            rna_scores = np.concatenate((rna_scores,h5py.File(f'{args.shap_dir}/{sf}','r')['rna_contrib_scores'][:,:,:]),axis=0)
    
    num_features = len(args.featurenames)
    for i in range(num_features):
        reg_score_dict = collections.defaultdict(dict) 
        rna_shap_all = rna_scores[:,:,i].flatten()
        for count in range(rna_scores.shape[0]):
            seq_info = seqs.iloc[count]
            for reg,reg_name in zip(args.regs,args.regnames):
                if(int(reg)==100):
                    sub_index = range(512)
                    rna_shap_reg = rna_scores[count,:,i]
                elif(int(reg)<40):
                    sub_per = np.percentile(rna_shap_all,int(reg)) 
                    sub_index = np.where(rna_scores[count,:,i]<=sub_per)[0]
                    rna_shap_reg = rna_scores[count,sub_index,i]
                elif(int(reg)>60):
                    sub_per = np.percentile(rna_shap_all,int(reg))
                    sub_index = np.where(rna_scores[count,:,i]>=sub_per)[0]
                    rna_shap_reg = rna_scores[count,sub_index,i]
                elif(int(reg==50)):
                    up_i = int(reg)-5
                    down_i = int(reg)+5
                    sub_per_1 = np.percentile(rna_shap_all,up_i)
                    sub_per_2 = np.percentile(rna_shap_all,down_i)
                    sub_index = np.where((rna_scores[count,:,i]>=sub_per_1) & (rna_scores[count,:,i]<=sub_per_2))[0]
                    rna_shap_reg = rna_scores[count,sub_index,i]
                
                for (index,rna_shap) in zip(sub_index,rna_shap_reg):
                    start = seq_info['start']+int(index)*2048
                    end = start +2048
                    binid = seq_info['chrom']+"\t"+str(start)+"\t"+str(end)
                    if(binid in reg_score_dict[reg_name]):
                        if(abs(rna_shap) > abs(reg_score_dict[reg_name][binid])):
                            reg_score_dict[reg_name][binid]=rna_shap
                    else:
                        reg_score_dict[reg_name][binid]=rna_shap
                
 
                fout= open(f'{args.outdir}/BinScores/{args.featurenames[i]}_{reg_name}_important_bins.txt', "w")
                fout.write("chrom\tstart\tend\t"+args.featurenames[i]+"_Shap\n") 
                for key in reg_score_dict[reg_name]:
                    fout.write(key+"\t"+str(reg_score_dict[reg_name][key])+"\n")


if __name__ == '__main__':
    main() 
