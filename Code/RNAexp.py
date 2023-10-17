#!/usr/bin/env python3

import io
import sys
import numpy as np
import pandas as pd
import cooler
from cooler.core import RangeSelector1D
import argparse

class ExpCount:
    def __init__(self,coolfile):
        self.coolfile = coolfile
        clr = cooler.Cooler(coolfile)
        DNA = pd.DataFrame(clr.bins()[:])
        self.DNAinfo = DNA.loc[:,'chrom':'end']
        self.n_bins = clr.info['nbins']
        self.pixels = clr.pixels()[:]
        self.weights = clr.bins()[['chrom','start','end']]
        self.annotate()
               
    def annotate(self):
        """
        Add bin annotations to a data frame of pixels. Modified from cooler.api.annotate
        """
        columns = self.pixels.columns
        ncols = len(columns)
        is_selector = isinstance(self.weights, RangeSelector1D)

        if "bin1_id" in columns:
            if is_selector:
               right = self.weights[:]
            else:
               right = self.weights
            self.pixels = self.pixels.merge(right, how="left", left_on="bin1_id", right_index=True)

        if "bin2_id" in columns:
            self.pixels = self.pixels.merge(right, how="left", left_on="bin2_id", right_index=True, suffixes=("1", "2"))

        self.pixels = self.pixels[list(self.pixels.columns[ncols:]) + list(self.pixels.columns[:ncols])]

    def get_nascent_RNA(self,outfile):
        marg_RNA = np.bincount(self.pixels['bin2_id'], weights=self.pixels['count'], minlength=self.n_bins)
        nascent_df = self.DNAinfo.copy(deep=True)
        nascent_df['sum'] = pd.Series(marg_RNA)
        nascent_df.to_csv(outfile,sep="\t",index=False)

    def get_DNA_attached_RNA(self,outfile):
        marg_RNA = np.bincount(self.pixels['bin1_id'], weights=self.pixels['count'], minlength=self.n_bins)
        RAL_df = self.DNAinfo.copy(deep=True)
        RAL_df['sum'] = pd.Series(marg_RNA)
        RAL_df.to_csv(outfile,sep="\t",index=False)

    def get_trans_RNA(self,rnafile,genefile,outfile):
        pixels_filter = self.pixels.copy(deep=True)
        longgenes = pd.read_table(genefile,sep='\t',header=0)
        for geneindex, generow in longgenes.iterrows():
            gene_mask = (pixels_filter['chrom1']==generow['chrom']) & (pixels_filter['start1']>generow['start']) & \
                        (pixels_filter['start1']<generow['end']) & (pixels_filter['chrom2']==generow['chrom']) & \
                        (pixels_filter['start2']>generow['start']) & (pixels_filter['start2']<generow['end'])
            pixels_filter['count'][gene_mask] = 0
        pixels_same_chrom = pixels_filter[pixels_filter["chrom1"] == pixels_filter["chrom2"]][['chrom1', 
                            'start1', 'end1', 'chrom2', 'start2', 'end2','bin1_id', 'bin2_id', 'count']]
        pixels_diff_chrom = pixels_filter[pixels_filter["chrom1"] != pixels_filter["chrom2"]][['chrom1', 
                            'start1', 'end1', 'chrom2', 'start2', 'end2','bin1_id', 'bin2_id', 'count']]
        pixels_same_chrom_ge1m = pixels_same_chrom[abs(pixels_same_chrom['bin1_id']-pixels_same_chrom['bin2_id'])>512]
        pixels_trans = pd.concat([pixels_same_chrom_ge1m, pixels_diff_chrom], ignore_index=True)
        
        RNA_type = pd.read_table(rnafile,header=0).rename(columns={'id': 'bin2_id'})
        pixels_trans_rnatype = (pixels_trans.merge(RNA_type, on=["bin2_id"], how="inner"))
        trans_df = self.DNAinfo.copy(deep=True)
        rnatypes = ['other_smallRNA','snRNA','snoRNA','misc_RNA','lncRNA','protein_coding','others','.']
        for rt in rnatypes:
            pixels_trans_sub = pixels_trans_rnatype[pixels_trans_rnatype['RNA_bin_annotation']==rt]
            marg_RNA = np.bincount(pixels_trans_sub['bin1_id'], weights=pixels_trans_sub['count'], minlength=self.n_bins)
            if (rt=="."):
                trans_df['nogene'] = pd.Series(marg_RNA)
            else:
                trans_df[rt] = pd.Series(marg_RNA)
        trans_df.to_csv(outfile,sep="\t",index=False)
