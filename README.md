# Exploring the Roles of RNAs in Chromatin Architecture Using Deep Learning
We investigate the roles of chromatin-associated RNAs (caRNAs) in genome folding in HFFc6 cells by bioinformatics analyses and proposing a deep learning framework (AkitaR) that leverages both genome sequences and genome-wide RNA-DNA interactions. 

## Installation
AkitaR is built on [Akita](https://github.com/calico/basenji/tree/master/manuscripts/akita). To run AkitaR, the installation of basenji/Akita is needed, which could be found at :[https://github.com/calico/basenji/tree/master]. A detailed list of required packages for AkitaR, data exploration, downstream analyses and visualization could also be found in the AkitaR.yml. The yml file could be used to create a virtual environemnt with all the required packages using [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Our codebase has been tested on Red Hat Enterprise Linux 8.9 using both CPU s and NVIDIA A30 GPUs. For model training, NVIDIA GPUs with CUDA are recommended.

## Instructions
Code for [AkitaR](/Code/AkitaR/) training and evaluation are under Code/AkitaR. 
### Preprocess datasets for AkitaR training
The inputs of AkitaR are the one-hot encoded sequences and additional RNA and/or ATAC-seq features, and the output is the contact maps. To prepare for the inputs and outputs, genome fasta file, data to get RNA features and binned Hi-C or Micro-C data in cooler format are needed. 

Here, we showed an example to extract the trans-located caRNA signals from binned RNA-chromatin data in cooler format, and then use the signals together with genome fasta file and Micro-C data to prepare the datasets for AkitaR training.

Code for extracting trans-located caRNA features:
```bash
python get_RNA_abundance_by_bin.py --infile HFFc6_iMARGI.cool --rnatype Data/Genomic_bin_data/Genomic_bin_2048_RNAtype_anno --longgene Data/Human_genome/gencode.v43.gene.length.ge1M.txt --output Data/Genomic_bin_data/HFFc6_RNA_class_count_long_cis_trans_2048
```
[Genomic_bin_2048_RNAtype_anno](/Data/Genomic_bin_data/Genomic_bin_2048_RNAtype_anno) and [gencode.v43.gene.length.ge1M.txt](/Data/Human_genome/gencode.v43.gene.length.ge1M.txt) could be found under Data folder. The output of the above code could be found at [HFFc6_RNA_class_count_long_cis_trans_2048](/Data/Genomic_bin_data/HFFc6_RNA_class_count_long_cis_trans_2048).

The information of the trans-located caRNA signals and Micro-C data is then stored in two files, as shown in [trans_RNA.txt](/Data/Model_prepartion/trans_RNA.txt) and [targets.txt](/Data/Model_prepartion/targets.txt).
Code for get the datasets for AkitaR training:
```bash
python akitaR_data.py -n 8 -b Data/Human_genome/hg38.blacklist.bed -g Data/Human_genome/hg38_gaps_binsize2048_numconseq10.bed -k 1 -l 1048576 --crop 65536 -o Data/Model_input --as_obsexp -p 6 -t .1 -v .1 -w 2048 --snap 2048 --stride_train 262144 --stride_test 524288 Data/Human_genome/hg38.ml.fa Data/Model_prepartion/trans_RNA.txt Data/Model_prepartion/targets.txt
```
[hg38.blacklist.bed](/Data/Human_genome/hg38.blacklist.bed) and [hg38_gaps_binsize2048_numconseq10.bed](/Data/Human_genome/hg38_gaps_binsize2048_numconseq10.bed) could be found under the Data folder. A sample output of the above code could be found under [Data/Model_input]

### Train/Test model
To train the model with the datasets generated above, please use the following code:
```bash
akitaR_train.py -o train_out Data/Model/params.json Data/Model_input
```
[params.json](/Data/Model/params.json) includes the hyperparametres used for model training.

To evaluate the model performance, the following code could be used:
```bash
akitaR_test.py --save -o test_out Data/Model/params.json train_out/model_best.h5 Data/Model_input
```
### Compute contribution scores of input features
After the model is trained and tested, the contributions of each feature could be calculated using the following codes:
```bash
python save_inputs.py -d Data/Model_input -p Data/Model/params.json -l test -n 8 -o inputs_dir
python get_feature_contribution_scores_with_DeepExplainer.py -d inputs_dir -p Data/Model/params.json -m train_out/model_best.h5 -l test -n 8 -o IS_output_dir
```

### Codes for data exploration, downstream analyses and visualization
- [Source code](/Code) Custom code for data exploration and downstream analyses.
- [Figures](/Code/CaRNAs_in_Chromatin_Architecture_Figures.ipynb) Jupyter notebook for generating the figures.

## Data
- [Data](/Data) Some of the data to demonstrate the use of the AkitaR code and data for generating the figures. More data could be found at: [https://zenodo.org/records/10015009].

## Contact
If you have any questions, please feel free to contact shuzhen.kuang@gladstone.ucsf.edu or szkuang@gmail.com.


