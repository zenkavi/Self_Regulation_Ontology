# simulate zero-inflated NB data
# compare ZINB regression to dichotomization + classification

library("mpath")
library("zic")
library("pscl")
library(glmnet)

rand_zinb=function(X,b1=2,b2=6,sd1=1,sd2=1){
  pcount=0
  while (sum(round(pcount))==0){
    beta1=array(0,dim=dim(X)[2])
    beta2=array(0,dim=dim(X)[2])
    beta1[1:2]=b1
    beta2[3:4]=b2
    a1 =X%*%beta1+rnorm(dim(X)[2])*sd1
    a2 =X%*%beta2+rnorm(dim(X)[2])*sd2
    pzero = exp(a2)/(1+exp(a2))
    pcount = exp(a1)*(1-pzero)
  }
  return(round(pcount))
}

shuffle=FALSE
nruns=2500
nobs=500
nvars=20
zinbcor=array(NA,dim=nruns)
bincor=array(NA,dim=nruns)
bincor_count=array(NA,dim=nruns)

for (r in 1:nruns) {
  print(sprintf('run %d',r))
  X<-matrix(runif(nobs*nvars), ncol=nvars) 
  
  pcount=rand_zinb(X)
  if (shuffle) {pcount=pcount[sample(length(pcount))]}
  
  nfolds=4
  nsets=ceiling(dim(X)[1]/nfolds)
  fold=kronecker(rep(1,nsets),seq(1,nfolds))[1:dim(X)[1]]
  fold=sample(fold)
  pred_count=array(NA,dim=dim(X)[1])
  pred_bin=array(NA,dim=dim(X)[1])
  pred_resp=array(NA,dim=dim(X)[1])
  for (f in 1:nfolds) {
    d=as.data.frame(X[fold!=f,])
    d$y=pcount[fold!=f]
    
    z=zipath(y~.|.,d,family='negbin',nlambda=10)
    bestaic=which(z$aic==min(z$aic))
    pred_count[fold==f]=predict(z,as.data.frame(X[fold==f,]),type='response',which=bestaic)
    l=cv.glmnet(X[fold!=f,],as.integer(pcount[fold!=f]>0),type.measure="auc",family='binomial')
    pred_bin[fold==f]=as.integer(predict(l,newx=X[fold==f,], s="lambda.min",type='class'))
    pred_resp[fold==f]=predict(l,newx=X[fold==f,], s="lambda.min",type='response')
  }
  zinbcor[r]=cor(pred_count,pcount)
  bincor[r]=cor(pred_bin,as.integer(pcount>0))
  bincor_count[r]=cor(pred_resp,pcount)
  print(sprintf('%d %f %f %f',r,zinbcor[r],bincor[r],bincor_count[r]))
}
if (shuffle) {
  save(zinbcor,bincor,bincor_count,file='shuf.Rdata')
} else {
  save(zinbcor,bincor,bincor_count,file='noshuf.Rdata')
}