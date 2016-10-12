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
    datafile <- paste0(basedir, 'Data', dataset, 'meaningful_variables_noEZ_contrasts.csv')
  }
  d <- read.csv(datafile, row.names=1)
  return(d)
}

get_info <- function(item,infile='../Self_Regulation_Settings.txt'){
  
  if(file.exists(infile)){
    
    infodict <- suppressWarnings(read.table(infile, sep = ":"))
    
    infodict$V2 <- gsub(" ", "", infodict$V2)
    
    if(item %in% infodict$V1){
      return(as.character(infodict[infodict$V1 == item, "V2"])) 
    }
    else {
      print('infodict does not include requested item')
    }
    
  }
  else{
    print('You must first create a Self_Regulation_Settings.txt file')
  }
}

