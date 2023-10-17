#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
import gzip
import collections
from utils import read_bed_regions

def main():
    parser.add_argument("--infile", "-i", help="RNA-DNA_bin interaction with FDR from hypergeo_test", required=True)
    parser.add_argument( "--bedfiles","-b", nargs='+', help='<Required> Set of selected regions', required=True)
    parser.add_argument( "--bednames","-a", nargs='+', help='<Required> Set of selected region names', required=True)
    parser.add_argument( "--thershold","-t",type=float,help="thershold for FDR values")
    parser.add_argument( "--outdir","-o",help="dir of output files", required=True)
    parser.add_argument( "--outnames","-n", nargs='+', help='<Required> Set of output file names', required=True)
    args = parser.parse_args()
    
    RNAs = collections.defaultdict(dict)
    RNA_levels = collections.defaultdict(dict)
    bed_regions = collections.defaultdict(dict)
    bed_nums = []
    for bn, bf in zip(args.bednames,args.bedfiles):
        bed_regions[bn], reg_num = read_bed_regions(bf,header=True,get_reg_num=True)
        bed_nums.append(reg_num)
    with open(infile, 'r') as f:
        for line in f:
            if('Geneinfo' in line):
                continue
            detail = line.strip().split('\t')
            g = detail[0]
            rid = '\t'.join([detail[1],detail[2],detail[3]])
            if(float(detail[9])<=args.thershold):
                for bn in bednames:
                    if(rid in bed_regions[bn]):
                        if(g in RNAs[bn]):
                            RNAs[bn][g]+=1
                        else:
                            RNAs[bn][g]=1
                        if(g in RNA_levels[bn]):
                            RNA_levels[bn][g]+=int(detail[4])
                        else:
                            RNA_levels[bn][g]=int(detail[4])
 
    for bname,bsize,oname in zip(args.bednames,bed_nums,args.outnames):  
        fout = open(f'{args.outdir}/{oname}', "w")
        fout.write('\t'.join(['Geneinfo','sum_inter_counts','mean_inter_counts','bin_counts','bin_ratio\n']))                          
        for key in RNA_levels[bname]:
            normalized = round((RNA_levels[bname][key]/bsize),5)
            count = RNAs[bname][key]
            ratio = count/bsize
            fout.write(key+"\t"+str(RNA_levels[bname][key])+"\t"+str(normalized)+"\t"+str(count)+"\t"+str(ratio)+"\n")
        fout.close()

if __name__ == '__main__':
    main()

