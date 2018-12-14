#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

#Usage:
#Rscript --vanilla bootstrap_retest.R data_dir out_dir n dv_name

#test if all arguments are supplied
# test if there is at least one argument: if not, return an error
if (length(args)<3) {
  stop("Arguments are missing. Usage: Rscript --vanilla bootstrap_retest.R t1_data t2_data out_dir n dv_name", call.=FALSE)
}

t1_data <- args[1]
t2_data <- args[2]
output_dir <- args[3]
n <- as.numeric(args[4])
dv_name <- args[5]

#load packages
library(tidyverse)
library(RCurl)
library(psych)

#load data
retest_subs_test_data <- read.csv(t1_data)
retest_data <- read.csv(t2_data)

names(retest_subs_test_data)[which(names(retest_subs_test_data) == "X")] <- "sub_id"
names(retest_data)[which(names(retest_data) == "X")] <- "sub_id"

#bootstrap 1000 times

file_names = c('sem.R', 'get_numeric_cols.R', 'match_t1_t2.R', 'get_retest_stats.R', 'make_rel_df.R')

helper_func_path = 'https://raw.githubusercontent.com/zenkavi/SRO_Retest_Analyses/master/code/helper_functions/'
for(file_name in file_names){
  eval(parse(text = getURL(paste0(workspace_scripts,file_name), ssl.verifypeer = FALSE)))
}

sample_workers = function(N = 150, repl= TRUE, df=retest_data, worker_col = "sub_id"){
  return(sample(df[,worker_col], N, replace = repl))
}

bootstrap_reliability = function(metric = c('icc', 'spearman','pearson', 'partial_eta_sq', 'eta_sq', 'omega_sq', 'sem', 'var_breakdown', 'aov_stats'), dv_var, worker_col="sub_id"){
  tmp_sample = sample_workers(worker_col = worker_col)
  out_df = data.frame(dv = dv_var)
  if('icc' %in% metric){
    out_df$icc = get_icc(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)
  }
  if('spearman' %in% metric){
    out_df$spearman = get_spearman(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)
  }
  if('pearson' %in% metric){
    out_df$pearson = get_pearson(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)
  }
  if('partial_eta_sq' %in% metric){
    out_df$partial_eta_sq = get_partial_eta(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)
  }
  if('eta_sq' %in% metric){
    out_df$eta_sq = get_eta(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)
  }
  if('omega_sq' %in% metric){
    out_df$omega_sq = get_omega(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)
  }
  if('sem' %in% metric){
    out_df$sem = get_sem(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)
  }
  if('var_breakdown' %in% metric){
    out_df$var_subs = get_var_breakdown(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)$subs
  }
  if('var_breakdown' %in% metric){
    out_df$var_ind = get_var_breakdown(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)$ind
  }
  if('var_breakdown' %in% metric){
    out_df$var_resid = get_var_breakdown(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)$resid
  }
  if('aov_stats' %in% metric){
    out_df$F_time = get_aov_stats(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)$F_time
  }
  if('aov_stats' %in% metric){
    out_df$p_time = get_aov_stats(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)$p_time
  }
  if('aov_stats' %in% metric){
    out_df$df_time = get_aov_stats(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)$df_time
  }
  if('aov_stats' %in% metric){
    out_df$df_resid = get_aov_stats(dv_var, sample = 'bootstrap', sample_vec = tmp_sample, merge_var = worker_col)$df_resid
  }
  return(out_df)
}

# generate random seed
cur_seed <- sample(1:2^15, 1)
set.seed(cur_seed)

# bootstrap given variable for given n times
output_df = plyr::rdply(n, bootstrap_reliability(dv_var = dv_name))

# add seed info
output_df$seed <- cur_seed

# save output
write.csv(output_df, paste0(output_dir, dv_name, '_output.csv'))
