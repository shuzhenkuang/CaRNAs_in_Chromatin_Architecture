#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
import gzip
from utils import get_data_dict,read_gtf,check_self_interactions,get_TPM

def main():
    parser = argparse.ArgumentParser()                                               
    parser.add_argument("--infile", "-f", help="RNA DNA interactions with gene annotations", required=True)
    parser.add_argument("--hostgenes", "-g", help="genes and their host genes", required=True)
    parser.add_argument("--self", "-s", default=False, action='store_true',help="remove self interactions of genes in the list")
    parser.add_argument("--genelist", "-l", help="gene list") 
    parser.add_argument("--norm", "-n", default=False, action='store_true',help="get TPM normalization of gene abundance")
    parser.add_argument( "--output","-o",help="gene abundance", required=True)
    args = parser.parse_args()
    
    hostgene = get_data_dict(args.hostgenes,header='HostGene')

    RNA_levels = {}
    with gzip.open(args.infile, "rt") as f:
        for line in f:
            if('chrom2' in line):
                continue
            detail = line.strip().split('\t')
            genes = detail[10].split(',')
            geneinfos = []
            for i in range(0,len(genes),2):
                if(genes[i]!="."):
                    gd = genes[i]+","+genes[i+1]
                geneinfos.append(gd)
            if(args.self):
                if(not args.genelist):
                    parser.error('Must provide the list of genes to remove their self interactions') 
                else:
                    genes_list = read_gtf(args.genelist, 'gene', 'gene_id,gene_name,gene_type', stranded = True, get_genes=True)
                skip = check_self_interactions(geneinfos,detail[11],genes_list)
                if(skip):
                    continue
            for g in geneinfos:
                if (g in RNA_levels):
                    RNA_levels[g] +=1
                else:
                    RNA_levels[g] =1
    
    for key, value in RNA_levels.items():
        if key in hostgene:
            subgene = hostgene[key].split(';')
            for sg in subgene:
                if(sg in RNA_levels):
                    value -= RNA_levels[sg]
   
    RNA_exp_df = pd.DataFrame.from_dict(RNA_levels,columns=['GeneInfo','Count'])
    if(args.norm):
        RNA_exp_df[['GeneID','GeneName','GeneType','GeneLoc']] = RNA_exp_df['GeneInfo'].str.split(pat = '|', expand = True)    
        RNA_exp_df['Start'] = RNA_exp_df['GeneLoc'].str.extract('\[(\d+),\d+')
        RNA_exp_df['End'] = RNA_exp_df['GeneLoc'].str.extract('\[\d+,(\d+)')
        RNA_exp_df['Length'] = RNA_exp_df['End'].astype(int)-RNA_exp_df['Start'].astype(int)
        RNA_exp_df_sub =  RNA_exp_df[['GeneID','Length','Count']]
        RNA_exp_norm = get_TPM(RNA_exp_df_sub)
        RNA_exp_norm_df = RNA_exp_df.merge(RNA_exp_norm, on=['GeneID'])
        RNA_exp_norm_df.to_csv(args.output,sep="\t",index=False)
    else:
        RNA_exp_df.to_csv(args.output,sep="\t",index=False)

if __name__ == '__main__':
    main()
    
