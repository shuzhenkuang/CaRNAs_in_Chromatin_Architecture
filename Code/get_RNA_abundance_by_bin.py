#!/usr/bin/env python3

import io
import sys
import numpy as np
import pandas as pd
import argparse
import RNAexp

def main():
    parser = argparse.ArgumentParser()                                               
    parser.add_argument("--infile", "-f", help="input cool file",required=True)
    parser.add_argument("--cis", "-c", default=False, action='store_true',help="get nascent transcriotion")
    parser.add_argument("-ral", "-a", default=False, action='store_true',help="get RNA attached to genomic bin")
    parser.add_argument("--rnatype", "-r", help="main RNA type transcribed from each bin")
    parser.add_argument("--longgene", "-l", help="genes longer than 1Mb")
    parser.add_argument( "--output","-o",help="output file",required=True)
    args = parser.parse_args()
   
    cool_data = RNAexp.ExpCount(args.infile)
    if(args.cis):
        cool_data.get_nascent_RNA(args.output)
    else:
        if(args.ral):
            cool_data.get_DNA_attached_RNA(args.output)
        else:
            if((args.rnatype) and (args.longgene)):
                cool_data.get_trans_RNA(args.rnatype,args.longgene,args.output)
            else:
                parser.error('Must provide rna type of each bin and information of long genes to get trans-attached RNA level')

if __name__ == '__main__':
    main()
