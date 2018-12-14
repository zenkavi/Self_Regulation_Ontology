#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

# t1_data <- args[1]
# t2_data <- args[2]
data_files <- args[1]
output_dir <- args[3]
n <- as.numeric(args[4])
# dv_name <- args[5]

#load packages
library(tidyverse)
library(RCurl)
library(psych)

#load data

workspace_scripts = 'https://raw.githubusercontent.com/zenkavi/SRO_Retest_Analyses/master/code/workspace_scripts/'

eval(parse(text = getURL(paste0(workspace_scripts,data_files), ssl.verifypeer = FALSE)))

#helper functions

file_names = c('sem.R', 'get_numeric_cols.R', 'match_t1_t2.R', 'get_retest_stats.R', 'make_rel_df.R')

helper_func_path = 'https://raw.githubusercontent.com/zenkavi/SRO_Retest_Analyses/master/code/helper_functions/'
for(file_name in file_names){
  eval(parse(text = getURL(paste0(workspace_scripts,file_name), ssl.verifypeer = FALSE)))
}

# boot function

bootstrap_reliability = function(t1_df=test_data, t2_df=retest_data, metrics = c('spearman', 'pearson', 'var_breakdown', 'partial_eta', 'sem','icc2.1', 'icc3.k'), dv_var, worker_col="sub_id"){
  
  indices = sample(1:150, 150, replace=T)
  sampled_test_data = test_data[indices,]
  sampled_retest_data = retest_data[indices,]
  
  Sys.time()
  out_df = make_rel_df(t1_df = sampled_test_data, t2_df = sampled_retest_data, metrics = metrics)
  Sys.time()
  
  return(out_df)
}

# generate random seed
cur_seed <- sample(1:2^15, 1)
set.seed(cur_seed)

# bootstrap sampled dataset for given n times
output_df = plyr::rdply(n, bootstrap_reliability())

# add seed info
output_df$seed <- cur_seed

# save output
write.csv(output_df, paste0(output_dir, 'bootstrap_output.csv'))
