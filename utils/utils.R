install.packages(c('caret', 'e1071'), repos = 'http://cran.us.r-project.org')
library(e1071)
library(caret)

print_confusion_matrix <- function(y_true, y_pred){
  return(caret::confusionMatrix(y_pred, y_true))
}

get_behav_data <- function(dataset, use_EZ=FALSE){
  basedir <- get_info('base_directory')
  if(use_EZ==T){
    datafile <- paste0(basedir, 'Data', dataset, 'meaningful_variables_EZ_contrasts.csv')
  }
  else {
    datafile <- (basedir,'Data',dataset,'meaningful_variables_noEZ_contrasts.csv')
  }
  d <- read.csv(datafile, row.names=1)
  return(d)
}

get_info <- function(){
  
}
