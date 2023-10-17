#!/usr/bin/env python3

import io
import sys
import numpy as np
import pandas as pd
import argparse
import gzip
import HTSeq
import csv
from collections import defaultdict
from utils import read_peak,get_overlap_regions,get_overlap_value

def main():
    parser = argparse.ArgumentParser()                                               
    parser.add_argument("--infile", "-f", help="bedpe gzip file of RNA DNA interactions",required=True)
    parser.add_argument("--annofile","-a", help="annotation states of genomic regions",required=True)
    parser.add_argument("--minover", "-m", type=float, help="mininum overlap ratio between regions",required=True)
    parser.add_argument( "--output","-o", help="stats of all RNA DNA interactions",required=True)
    parser.add_argument( "--cisoutput","-c",help="stats of RNA DNA interactions on the same chromosome",required=True)
    parser.add_argument( "--transoutput","-t",help="stats of RNA DNA interactions on different chromosomes",required=True)
    args = parser.parse_args()
 
    fout = open(args.output, "w")
    fcisout = open(args.cisoutput, "w")
    ftransout = open(args.transoutput, "w")
    min_over = float(args.minover)
    stranded = False
    annotations = read_peak(args.annofile,stranded)
    interactions = {}
    cis_interactions = {}
    trans_interactions = {}
    
    with gzip.open(args.infile, "rt") as f:
        for line in f:
            if('chrom2' in line):
                continue
            detail = line.strip().split('\t')
            chrom1 = detail[0]
            start1 = int(detail[1])
            end1 = int(detail[2])
            size1 = end1 - start1
            chrom2 = detail[3]
            start2 = int(detail[4])
            end2 = int(detail[5])
            size2 = end2 - start2
            region1_iv = HTSeq.GenomicInterval(chrom1, start1, end1, '.')
            region2_iv = HTSeq.GenomicInterval(chrom2, start2, end2, '.')
            annotation_regions1 = get_overlap_regions(region1_iv,annotations,size1,min_over)
            annotation_regions2 = get_overlap_regions(region2_iv,annotations,size2,min_over)
            max_value1 = get_overlap_value(annotation_regions1)
            max_value2 = get_overlap_value(annotation_regions2)
            if((max_value1 != "-") and (max_value2 != "-")):
                inter_id = max_value1+"\t"+max_value2
                if(inter_id in interactions):
                    interactions[inter_id] +=1
                else:
                    interactions[inter_id] =1
                if(chrom1==chrom2):
                    if(inter_id in cis_interactions):
                        cis_interactions[inter_id] +=1
                    else:
                        cis_interactions[inter_id] =1
                else:
                    if(inter_id in trans_interactions):
                        trans_interactions[inter_id] +=1
                    else:
                        trans_interactions[inter_id] =1
          
    for ikey in interactions:  
         fout.write(ikey+"\t"+str(interactions[ikey])+"\n")
    for ckey in cis_interactions:
         fcisout.write(ckey+"\t"+str(cis_interactions[ckey])+"\n")
    for tkey in trans_interactions:
         ftransout.write(tkey+"\t"+str(trans_interactions[tkey])+"\n")

if __name__ == '__main__':
    main()

