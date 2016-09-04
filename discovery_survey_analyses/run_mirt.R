library(mirt)
library(parallel)
mirtCluster(23)
args <- commandArgs(TRUE)

if (length(args)<1) {
ncomps=7
} else {
  ncomps=as.integer(args[1])
}


d=read.csv('surveydata.csv')
d$worker=NULL
m=mirt(d,ncomps,verbose=FALSE,method='MHRM')

s=summary(m)

scores=s$rotF

vnames=read.csv('variable_key.txt',sep='\t',header=FALSE)

for (i in 1:ncomps) {
  s=sort(scores[,i])
  sd=sort(scores[,i],decreasing=TRUE)
  n=names(s)
  nd=names(sd)
  for (j in 1:3){
    
    cat(s[j],n[j],as.character(vnames[as.character(vnames$V1)==n[j],]$V2), "\n")
  }
  for (j in 1:3){
    
    cat(sd[j],nd[j],as.character(vnames[as.character(vnames$V1)==nd[j],]$V2), "\n")
  }
  cat('\n')
}
save(m,file=sprintf('mirt_%ddims.Rdata',ncomps))
