
def run_group_prediction(results, shuffle=False, classifier='lasso',
                       include_raw_demographics=False, rotate='oblimin',
                       verbose=False):

    
    
    if verbose:
        print('*'*79)
        print('Running Prediction, shuffle: %s, classifier: %s' % (shuffle, classifier))
        print('*'*79)
        factor_scores = self.EFA.get_scores(rotate=rotate)
        demographic_factors = self.DA.reorder_factors(self.DA.get_scores())
        c = factor_scores.shape[1]
        # get raw data reorganized by clustering
        clustering=self.HCA.results['EFA%s_%s' % (c, rotate)]
        labels = clustering['clustered_df'].columns
        raw_data = self.data[labels]
        
        targets = [('demo_factors', demographic_factors)]
        if include_raw_demographics:
            targets.append(('demo_raw', self.demographics))
        for target_name, target in targets:
            for predictors in [('EFA%s_%s' % (c, rotate), factor_scores), ('raw', raw_data)]:
                # predicting using best EFA
                if verbose: print('**Predicting using %s**' % predictors[0])
                run_prediction(predictors[1], 
                               target, 
                               self.get_output_dir(),
                               outfile='%s_%s_prediction' % (predictors[0], target_name), 
                               shuffle=shuffle,
                               classifier=classifier, 
                               verbose=verbose)
