library(dplyr)

test_data_path = '/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Complete_02-03-2018/'

retest_data_path = '/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_02-03-2018/'

t1_tbt = read.csv(gzfile(paste0(test_data_path,'Individual_Measures/threebytwo.csv.gz'), 'rt'))

t2_tbt = read.csv(gzfile(paste0(retest_data_path,'Individual_Measures/threebytwo.csv.gz'), 'rt'))

#get only retest subjects
t1_tbt = t1_tbt[as.character(t1_tbt$worker_id) %in% as.character(test_data$sub_id),]
t2_tbt = t2_tbt[as.character(t2_tbt$worker_id) %in% as.character(retest_data$sub_id),]

#dv's
calc_break_dvs = function(df, breaks=c(seq(0, 440, 10)[c(-1, -45)], 439)){
  
  df = df %>%
    filter(exp_stage != "practice")
  
  calc_dvs = function(df){
    
    df = df %>%
      mutate(correct_shift = lag(correct),
             task_switch_shift = lag(task_switch))
    
    missed_percent = mean(df$rt == -1)
    
    df = df %>%
      filter(rt != -1)
    
    df_correct = df %>%
      filter(correct == "True" & correct_shift == "True") 
    
    dvs = data.frame(acc = mean(ifelse(df$correct == "True",1,0)),
                     avg_rt_error = median(df$rt[df$correct == "False"]),
                     std_rt_error = sd(df$rt[df$correct == "False"]),
                     avg_rt = median(df_correct$rt),
                     std_rt = sd(df_correct$rt),
                     missed_percent = missed_percent)
    
    cue_switch_cost_rt_df = df_correct %>% group_by(CTI, cue_switch) %>% summarise(cue_switch_cost = median(rt)) %>% complete(cue_switch, CTI) %>% filter(cue_switch %in% c("stay", "switch")) %>% ungroup() %>% spread(cue_switch,cue_switch_cost) %>% group_by(CTI) %>% summarise(cue_switch_cost_rt = switch-stay)
    
    dvs$cue_switch_cost_rt_100 = cue_switch_cost_rt_df$cue_switch_cost_rt[cue_switch_cost_rt_df$CTI == 100]
    dvs$cue_switch_cost_rt_900 = cue_switch_cost_rt_df$cue_switch_cost_rt[cue_switch_cost_rt_df$CTI == 900]
    
    task_switch_rt = df_correct %>% mutate(task_switch = factor(ifelse(grepl("switch",task_switch), T, F))) %>% group_by(CTI, task_switch) %>% summarise(rt = median(rt)) %>% complete(task_switch, CTI) %>% ungroup() %>% filter(task_switch == T)
    
    cue_switch_rt = df_correct %>% group_by(CTI, cue_switch) %>% summarise(rt = median(rt)) %>% complete(cue_switch, CTI) %>% ungroup() %>% filter(cue_switch == "switch")
    
    dvs$task_switch_cost_rt_100 = task_switch_rt$rt[task_switch_rt$CTI == 100] - cue_switch_rt$rt[cue_switch_rt$CTI == 100]
    
    dvs$task_switch_cost_rt_900 = task_switch_rt$rt[task_switch_rt$CTI == 900] - cue_switch_rt$rt[cue_switch_rt$CTI == 900]
    
    cue_switch_cost_acc_df = df %>% group_by(CTI, cue_switch) %>% summarise(acc = mean(ifelse(correct == "True",1, 0))) %>% complete(cue_switch, CTI) %>% filter(cue_switch %in% c("stay", "switch")) %>% spread(cue_switch, acc) %>% ungroup() %>% group_by(CTI) %>% summarise(cue_switch_cost_acc = switch-stay)
    
    dvs$cue_switch_cost_acc_100 = cue_switch_cost_acc_df$cue_switch_cost_acc[cue_switch_cost_acc_df$CTI == 100]
    dvs$cue_switch_cost_acc_900 = cue_switch_cost_acc_df$cue_switch_cost_acc[cue_switch_cost_acc_df$CTI == 900]
    
    task_switch_acc = df %>% mutate(task_switch = factor(ifelse(grepl("switch",task_switch), T, F))) %>% group_by(CTI, task_switch) %>% summarise(acc = mean(ifelse(correct == "True",1,0))) %>% complete(task_switch, CTI) %>%  ungroup() %>% filter(task_switch == T)
    
    cue_switch_acc = df %>% group_by(CTI, cue_switch) %>% summarise(acc = mean(ifelse(correct == "True",1,0))) %>% complete(cue_switch, CTI) %>% ungroup() %>% filter(cue_switch == "switch")
    
    dvs$task_switch_cost_acc_100 = task_switch_acc$acc[task_switch_acc$CTI == 100] - cue_switch_acc$acc[cue_switch_acc$CTI == 100]
    
    dvs$task_switch_cost_acc_900 = task_switch_acc$acc[task_switch_acc$CTI == 900] - cue_switch_acc$acc[cue_switch_acc$CTI == 900]
    
    return(dvs)
  }
  
  out = data.frame()
  
  for(i in 1:length(breaks)){
    tmp = df[1:breaks[i],]
    dvs = calc_dvs(tmp)
    out = rbind(out, dvs)
  }
  
  # due to timeout trials exclusion not all subjects might have 360 rows
  # this would lead to na's for later trial breaks
  # first remove these
  # out = out %>% drop_na()
  
  # then calculate the dv's for the remaining trials
  if(nrow(df)!=440 & nrow(df)%%10 != 0){
    dvs = calc_dvs(df)
    out = rbind(out, dvs)
  }
  
  #to keep track for later merge
  out$breaks = row.names(out)
  
  return(out)
}

#get dv's for t1 and t2
t1_dvs = t1_tbt %>%
  group_by(worker_id) %>%
  do(calc_break_dvs(.)) %>%
  rename(sub_id = worker_id)

write.csv(t1_dvs, paste0(retest_data_path, 't1_dvs.csv'))
