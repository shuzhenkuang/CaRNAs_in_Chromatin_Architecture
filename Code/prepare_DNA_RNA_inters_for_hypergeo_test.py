#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
import gzip
import math
from utils import get_data_dict, read_bed_regions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", "-f", help="RNA DNA interactions with gene annotations", required=True)
    parser.add_argument("--bedfile", "-b", help="genomic bins", required=True)
    parser.add_argument("--hostgene", "-g", help="genes and their host genes", required=True)
    parser.add_argument("--genelist", "-l", help="selected gene list", required=True) 
    parser.add_argument( "--chromsize","-c",help="chromsome size", required=True)
    parser.add_argument( "--geneexp","-e",help="RNA abundance (trans-located)", required=True)
    parser.add_argument( "--transRNA","-t",help="trans-located RNA at the genomic bin", required=True)
    parser.add_argument( "--output","-o",help="gene abundance", required=True)
    args = parser.parse_args()

    bed_regions = read_bed_regions(args.bedfile)
    genes_list = read_gtf(args.genelist, 'gene', 'gene_id,gene_name,gene_type', stranded=True, get_genes=True)
    chr_size = get_data_dict(args.chromsize,header='chrom')
    hostgenes = get_data_dict(args.hostgene,header='HostGene')
    trans_RNA,totalsum = read_bed_regions(args.transRNA,header=True,get_sum=True)
    RNA_exp = get_data_dict(args.geneexp,header='Gene')
    fout = open(args.output, "w")
    fout.write('\t'.join(['Geneinfo','chrom','start','end','intercount','genecount','DNAcount','total','\n']))   
 
    RNAs_DNAs = {}
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
            skip = check_self_interactions(geneinfos,detail[11],genes_list)
            if(skip):
                continue
            chrom = detail[3]
            start = int(detail[4])
            end = int(detail[5])
            bin_s = math.floor(start/2048)
            bin_e = math.floor(end/2048)
            rc = chrom
            if(bin_s == bin_e):
                rs = bin_s*2048
                if((bin_s+1)*2048<=int(chr_size[chrom])):
                    re = (bin_s+1)*2048
                else:
                    re = int(chr_size[chrom])
            else:
                overlap1 = (bin_s+1)*2048-start
                overlap2 = end-bin_e*2048
                if(overlap1>=overlap2):
                    rs = bin_s*2048
                    re = (bin_s+1)*2048
                else:
                    rs = bin_e*2048
                    if((bin_e+1)*2048<=int(chr_size[chrom])):
                        re = (bin_e+1)*2048
                    else:
                        re = int(chr_size[chrom])
            rid ='\t'.join([rc,str(rs),str(re)])
            if(rid in bed_regions):
                for g in geneinfos:
                    inter_id = g+"\t"+rid
                    if (inter_id in RNAs_DNAs):
                        RNAs_DNAs[inter_id] +=1
                    else:
                        RNAs_DNAs[inter_id] =1

    for keyid in RNAs_DNAs:
        key_infos = keyid.split('\t')
        key_value = int(RNAs_DNAs[keyid])
        if(key_infos[0] in hostgenes):
            key_new_value = int(RNAs_DNAs[keyid])
            subgene = hostgenes[key_infos[0]].split(';')
            for sg in subgene:
                subgeneid = '\t'.join([sg,key_infos[1],key_infos[2],key_infos[3]])
                if(subgeneid in RNAs_DNAs):
                    key_new_value -= int(RNAs_DNAs[subgeneid])
            key_value = key_new_value
        if(key_value>0):
            DNAid = '\t'.join([key_infos[1],key_infos[2],key_infos[3]])
            Nv = RNA_exp[key_infos[0]]
            nv = trans_RNA[DNAid]
            kv = key_value
            Mv = totalsum
            fout.write(keyid+"\t"+str(kv)+"\t"+str(Nv)+"\t"+str(nv)+"\t"+str(Mv)+"\n")
            
if __name__ == '__main__':
    main()
