#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

#Usage:
#Rscript --vanilla bootstrap_retest.R data_dir out_dir n dv_name

#test if all arguments are supplied
# test if there is at least one argument: if not, return an error
if (length(args)<3) {
  stop("Arguments are missing. Usage: Rscript --vanilla bootstrap_retest.R data_dir t1_df_name t2_df_name out_dir n dv_name", call.=FALSE)
} 

data_dir <- args[1]
t1_df_name <- args[2]
t2_df_name <- args[3]
output_dir <- args[4]
n <- as.numeric(args[5])
dv_name <- args[6]

#load packages
library(dplyr)
library(tidyr)
library(psych)


#load data
retest_subs_test_data <- read.csv(paste0(data_dir, t1_df_name))
retest_data <- read.csv(paste0(data_dir, t2_df_name))

#bootstrap 1000 times

match_t1_t2 <- function(dv_var, t1_df = retest_subs_test_data, t2_df = retest_data, merge_var = 'sub_id', format = "long", sample = 'full', sample_vec){

  if(sample == 'full'){
    df = merge(t1_df[,c(merge_var, dv_var)], t2_df[,c(merge_var, dv_var)], by = merge_var)
  }
  else{
    df = merge(t1_df[t1_df[,merge_var] %in% sample_vec, c(merge_var, dv_var)], t2_df[t2_df[,merge_var] %in% sample_vec, c(merge_var, dv_var)],
               by=merge_var)
  }

  df = df %>%
    na.omit()%>%
    gather(dv, score, -sub_id) %>%
    mutate(time = ifelse(grepl('\\.x', dv), 1, ifelse(grepl('\\.y', dv), 2, NA))) %>%
    separate(dv, c("dv", "drop"), sep='\\.([^.]*)$') %>%
    select(-drop)


  if(format == 'wide'){
    df = df%>% spread(time, score)
  }

  return(df)
}

get_spearman = function(dv_var, t1_df = retest_subs_test_data, t2_df = retest_data, merge_var = 'sub_id', sample='full', sample_vec){

  if(sample=='full'){
    df = match_t1_t2(dv_var, t1_df = t1_df, t2_df = t2_df, merge_var = merge_var, format='wide')
  }
  else if(sample=='bootstrap'){
    df = match_t1_t2(dv_var, t1_df = t1_df, t2_df = t2_df, merge_var = merge_var, format='wide', sample='bootstrap', sample_vec = sample_vec)
  }

  rho = cor(df$`1`, df$`2`, method='spearman')

  return(rho)
}

get_pearson = function(dv_var, t1_df = retest_subs_test_data, t2_df = retest_data, merge_var = 'sub_id', sample='full', sample_vec){

  if(sample=='full'){
    df = match_t1_t2(dv_var, t1_df = t1_df, t2_df = t2_df, merge_var = merge_var, format='wide')
  }
  else if(sample=='bootstrap'){
    df = match_t1_t2(dv_var, t1_df = t1_df, t2_df = t2_df, merge_var = merge_var, format='wide', sample='bootstrap', sample_vec = sample_vec)
  }

  r = cor(df$`1`, df$`2`, method='pearson')

  return(r)
}

get_icc <- function(dv_var, t1_df = retest_subs_test_data, t2_df = retest_data, merge_var = 'sub_id', sample='full', sample_vec){
  if(sample=='full'){
    df = match_t1_t2(dv_var, t1_df = t1_df, t2_df = t2_df, merge_var = merge_var, format='wide')
  }
  else if(sample=='bootstrap'){
    df = match_t1_t2(dv_var, t1_df = t1_df, t2_df = t2_df, merge_var = merge_var, format='wide', sample='bootstrap', sample_vec = sample_vec)
  }

  df = df %>% select(-dv, -sub_id)
  icc = ICC(df)
  icc_3k = icc$results['Average_fixed_raters', 'ICC']
  return(icc_3k)
}

get_eta <- function(dv_var, t1_df = retest_subs_test_data, t2_df = retest_data, merge_var = 'sub_id', sample='full', sample_vec){
  if(sample=='full'){
    df = match_t1_t2(dv_var, t1_df = t1_df, t2_df = t2_df, merge_var = merge_var)
  }
  else if(sample=='bootstrap'){
    df = match_t1_t2(dv_var, t1_df = t1_df, t2_df = t2_df, merge_var = merge_var, sample='bootstrap', sample_vec = sample_vec)
  }

  mod = summary(aov(score~Error(sub_id)+time, df))
  ss_time = as.data.frame(unlist(mod$`Error: Within`))['Sum Sq1',]
  ss_error = as.data.frame(unlist(mod$`Error: Within`))['Sum Sq2',]
  eta = ss_time/(ss_time+ss_error)
  return(eta)
}

get_sem <- function(dv_var, t1_df = retest_subs_test_data, t2_df = retest_data, merge_var = 'sub_id', sample='full', sample_vec){
  if(sample=='full'){
    df = match_t1_t2(dv_var, t1_df = t1_df, t2_df = t2_df, merge_var = merge_var)
  }
  else if(sample=='bootstrap'){
    df = match_t1_t2(dv_var, t1_df = t1_df, t2_df = t2_df, merge_var = merge_var, sample='bootstrap', sample_vec = sample_vec)
  }
  mod = summary(aov(score~Error(sub_id)+time, df))
  ms_error = as.data.frame(unlist(mod$`Error: Within`))['Mean Sq2',]
  sem = sqrt(ms_error)
  return(sem)
}

sample_workers = function(N = 150, repl= TRUE, df=retest_data, worker_col = "sub_id"){
  return(sample(df[,worker_col], N, replace = repl))
}

bootstrap_relialibility = function(metric = c('icc', 'spearman','pearson', 'eta_sq', 'sem'), dv_var){
  tmp_sample = sample_workers()
  out_df = data.frame(dv = dv_var)
  if('icc' %in% metric){
    out_df$icc = get_icc(dv_var, sample = 'bootstrap', sample_vec = tmp_sample)
  }
  if('spearman' %in% metric){
    out_df$spearman = get_spearman(dv_var, sample = 'bootstrap', sample_vec = tmp_sample)
  }
  if('pearson' %in% metric){
    out_df$spearman = get_spearman(dv_var, sample = 'bootstrap', sample_vec = tmp_sample)
  }
  if('eta_sq' %in% metric){
    out_df$eta_sq = get_eta(dv_var, sample = 'bootstrap', sample_vec = tmp_sample)
  }
  if('sem' %in% metric){
    out_df$sem = get_sem(dv_var, sample = 'bootstrap', sample_vec = tmp_sample)
  }
  return(out_df)
}

# generate random seed
cur_seed <- sample(1:2^15, 1)
set.seed(cur_seed)

# bootstrap given variable for given n times
output_df = plyr::rdply(n, bootstrap_relialibility(dv_var = dv_name))

# add seed info
output_df$seed <- cur_seed

# save output
write.csv(output_df, paste0(output_dir, dv_name, '_output.csv'))

