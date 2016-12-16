

from statsmodels.regression.linear_model import OLS
from sklearn.model_selection import KFold
import numpy as N


class BalancedKFold:
    """
    This function uses anova across CV folds to find
    a set of folds that are balanced in their distriutions
    of the X value - see Kohavi, 1995
    - we don't actually need X but we take it for consistency
    """
    def __init__(self,nfolds=5,pthresh=0.8,verbose=False):
        self.nfolds=nfolds
        self.pthresh=pthresh
        self.verbose=verbose

    def split(self,X,Y):
        """
        - we don't actually need X but we take it for consistency
        """

        nsubs=len(Y)

        # cycle through until we find a split that is good enough
        cv=KFold(n_splits=self.nfolds,shuffle=True)

        good_split=0
        while good_split==0:
            ctr=0
            idx=N.zeros((nsubs,self.nfolds)) # this is the design matrix
            folds=[]
            for train,test in cv.split(Y):
                idx[test,ctr]=1
                ctr+=1
                folds.append([train,test])

            lm_y=OLS(Y-N.mean(Y),idx).fit()

            if lm_y.f_pvalue>self.pthresh:
                if self.verbose:
                    print(lm_y.summary())
                return folds

if __name__=="__main__":
    Y=N.random.randn(100,1)
    bf=BalancedKFold(4,verbose=True)
    s=bf.split(Y)
