#!/usr/bin/env python3

import numpy as np
import pandas as pd
import HTSeq
import csv
from bioinfokit.analys import norm

def read_peak(peak_file, stranded = True, name=None):
    regs = HTSeq.GenomicArrayOfSets("auto", stranded = stranded)
    for peak_info in \
            csv.reader(open(peak_file), delimiter="\t"):
        chrom = peak_info[0]
        start = peak_info[1]
        end = peak_info[2]
        if(name):
            regname = name
        else:
            regname = peak_info[3]
        strand = '.'
        length = int(end)-int(start)
        iv = HTSeq.GenomicInterval(chrom, int(start), int(end), strand)
        regs[iv] += str(regname)+"|"+str(length)
    return regs

def read_gtf(ant_file, ant_level, ant_attr, stranded = True, get_genes=False):
    gene_regions = {}
    geneinfos = {}
    gtf = HTSeq.GFF_Reader(ant_file)
    ant = HTSeq.GenomicArrayOfSets("auto", stranded = stranded)
    ant_attr = ant_attr.split(',')
    for feature in gtf:
        if feature.type == ant_level:
            attrs = []
            for i in ant_attr:
                attrs += [feature.attr[i]]
            attrs += [str(feature.iv)]
            ant[feature.iv] += "|".join(attrs)
            gene_regions[str(feature.iv)] = "|".join(attrs)
            gene_key = "|".join(attrs)
            geneinfos[gene_key] = ""
    if(get_genes):
        return geneinfos
    else:
        return ant, gene_regions

def get_overlap_regions(reg_iv,peak_regs,size,ratio_ther):
    anno_regs = set()
    for iv, val in peak_regs[reg_iv].steps():
        if(len(list(val))>0):
            detail = list(val)[0].split('|')
            overlap_ratio1 = int(iv.length)/size
            overlap_ratio2 = int(iv.length)/int(detail[1])
            if(overlap_ratio1 >= ratio_ther or overlap_ratio2 >= ratio_ther):
                anno_regs.add(detail[0]+","+str(overlap_ratio1)+","+str(overlap_ratio2))
    return anno_regs

def get_overlap_value(annotation_regs,get_all=False):
    max_value = '-'
    all_values = '-'
    if len(annotation_regs) > 1:
        cans = {}
        for candidate in annotation_regs:
            (can_name,can_over1,can_over2) = candidate.split(',')
            if(can_name in cans):
                cans[can_name] +=float(can_over1)
            else:
                cans[can_name] =float(can_over1)
        all_values = ""
        for c_key in cans:
            all_values += c_key+":"+str(cans[c_key])+";"
        max_value = max(cans, key=cans.get)
    elif(len(annotation_regs) ==1):
        max_value = list(annotation_regs)[0].split(',')[0]
        all_values = list(annotation_regs)[0].split(',')[0] + ":" + str(list(annotation_regs)[0].split(',')[1])
    if(get_all):
        return max_value,all_values
    else:
        return max_value

def get_data_dict(infile,header='chrom'):
    data_dict = {}
    with open(infile, "r") as source:
        for line in source:
            if(header in line):
                continue
            info = line.strip().split('\t')
            data_dict[info[0]] = info[1]
    return data_dict

def read_bed_regions(bedfile,header=False,get_sum=False,get_reg_num=False):
    regions ={}
    count = 0
    with open(bedfile, 'r') as source:
        for i,line in enumerate(source):
            if(header):
                if(i<1):
                    continue
            info = line.rstrip().split('\t')
            bid = '\t'.join([info[0],info[1],info[2]])
            if(get_sum):
               count+=int(float(info[3]))
               regions[bid] = int(float(info[3]))
            else:
               regions[bid]=""
    if(get_sum):
        return regions,count
    else:
        if(get_reg_num):
            return regions,i
        else:
            return regions

def get_TPM(df):
    df = df.set_index('GeneID')
    nm = norm()
    nm.tpm(df=df, gl='Length')
    tpm_df = nm.tpm_norm
    tpm_df.reset_index(inplace=True)
    tpm_df = tpm_df.rename(columns = {'index':'GeneID','Count':'iMARGI_TPM'})
    return tpm_df

def check_self_interactions(geneinfos,geneanno2,genelist):
    self_inter = False
    gene2details = geneanno2.split(',')
    gene2infos = []
    for j in range(0,len(gene2details),2):
        if(gene2details[j]=="."):
            gd2 = gene2details[j]+","+gene2details[j+1]
            gene2infos.append(gd2)
    gene_intersects = list(set(geneinfos) & set(gene2infos))
    for gi in gene_intersects:
        if(gi in genelist):
            self_inter = True
    return self_inter

