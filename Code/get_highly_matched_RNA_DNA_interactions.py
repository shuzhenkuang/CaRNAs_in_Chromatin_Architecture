#!/usr/bin/env python3

import numpy as np
import pandas as pd
import gzip
import pysam
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import argparse

def main():
    parser = argparse.ArgumentParser()                                               
    parser.add_argument("--interactions", "-i", help="RNA DNA interactions", required=True)
    parser.add_argument("--genome", "-g", help="hg38 human genome fasta file", required=True)
    parser.add_argument( "--output","-o", help="highly matched RNA DNA interactions", required=True)
    args = parser.parse_args()

    fasta_open = pysam.Fastafile(args.genome)
    fout = open(arag.output, "w")
    with gzip.open(args.interactions,'rt') as source:
        for line in source:
            if('chrom' in line):
                continue
            info = line.strip().split('\t')
            myseq_str1 = f'{info[0]}:{info[1]}-{info[2]} {info[8]}'
            myseq_str2 = f'{info[3]}:{info[4]}-{info[5]} {info[9]}'
            seq1 = fasta_open.fetch(info[0],int(info[1]),int(info[2])).upper()
            seq2 = fasta_open.fetch(info[3],int(info[4]),int(info[5])).upper()
            if(info[8]=="-"):
                seq1 = Seq(seq1).reverse_complement()
            if(info[9]=="-"):
                seq2 = Seq(seq2).reverse_complement()
            alignment = pairwise2.align.localxx(str(seq1), str(seq2),one_alignment_only=True)[0]
            max_len = get_max_match_length(list(alignment))
            len1 = len(seq1)
            len2 = len(seq2)
            score = list(alignment)[2]
            ratio1 = float(score)/len1
            ratio2 = float(score)/len2
            if((max_len>=10) and ((ratio1>=0.8) or (ratio2>=0.8)):
                fout.write(line.strip()+"\t"+str(max_len)+"\t"+str(score)+"\t"+str(ratio1)+"\t"+str(ratio2)+"\n")
                fout.write(format_alignment(*alignment))

def get_max_match_length(alignment):
    max_match_len_list = []
    max_match_len =0 
    match_index =0

    for i, (a, b) in enumerate(zip(alignment[0], alignment[1])):
        if a == b:
            if i == match_index:
                max_match_len +=1
                match_index = i+1
            else:
                match_index = i+1
                max_match_len_list.append(max_match_len)
                max_match_len =1
    max_max_match_len = max(max_match_len_list)
    return max_max_match_len

if __name__ == '__main__':
    main()
