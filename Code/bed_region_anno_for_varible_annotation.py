#!/usr/bin/env python3

import io
import sys
import re
import HTSeq
import csv
import numpy as np
import pandas as pd
import argparse
from utils import read_peak,get_overlap_regions,get_overlap_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bedfile", "-b", help="genomic regions for annotations", required=True)
    parser.add_argument("--annonames","-a", nargs='+', help='<Required> Set of annotation names', required=True)
    parser.add_argument("--filenames","-f", nargs='+', help='<Required> Set of annotation files', required=True)
    parser.add_argument("--annotypes","-t", nargs='+', help='<Required> Set of types of the annotations(count,value,detail)', required=True)
    parser.add_argument("--minover", "-m", type=float, help="num overlap ratio between regions",required=True)
    parser.add_argument( "--output","-o",help="annotated genomic regions", required=True)
    args = parser.parse_args()

    fout = open(args.output, "w")
    fout.write("chrom\tstart\tend")
    for key in args.annonames:
        fout.write("\t"+str(key))
    fout.write("\n")

    min_over = float(args.minover)
    stranded = False
    annotations = {}
    for(anno_name,file_name,anno_type) in zip(args.annonames,args.filenames,args.annotypes):
        if(anno_type == "count"):
            annotations[anno_name] = read_peak(file_name,stranded,anno_name)
        elif((anno_type == "value") or (anno_type=="detail")) :
            annotations[anno_name] = read_peak(file_name,stranded)
    
    with open(args.bedfile,'r') as f:
        for line in f:
            if('chrom' in line):
                continue
            reg = line.strip().split('\t')
            chrom=reg[0]
            start = int(float(reg[1]))
            end = int(float(reg[2]))
            size = abs(end-start)
            region_iv = HTSeq.GenomicInterval(chrom, start, end, '.')
            annotation_regions ={}
            for a_name in args.annonames:
                annotation_regions[a_name] = get_overlap_regions(region_iv,annotations[a_name],size,min_over)
            fout.write(chrom+"\t"+str(start)+"\t"+str(end))
            for(a_name,anno_type) in zip(args.annonames,args.annotypes):
                if(anno_type == "count"):
                    a_count = len(list(annotation_regions[a_name]))
                    fout.write("\t"+str(a_count))
                elif((anno_type == "value") or (anno_type== "detail")):
                    max_value,all_values = get_overlap_value(annotation_regions[a_name],get_all=True)
                    if(anno_type == "value"):
                        fout.write("\t"+str(max_value))
                    elif(anno_type == "detail"):
                        fout.write("\t"+str(all_values))
            fout.write("\n") 
            
if __name__ == '__main__':
    main()

