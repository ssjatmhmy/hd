from util import saveit, loadit, dumpit, submit
from estimator import (RFREstimator, XGBEstimator, LassoEstimator, RidgeEstimator, 
                        KNNEstimator, LSVREstimator, KernelRidgeEstimator)
from hd_metrics import fmean_squared_error
import numpy as np


class EnsembleSelection(object):
    """
    Ref: Ensemble Selection from Libraries of Models. Caruana Rich, Niculescu-Mizil Alexandru, 
    Crew Geoff. ICML 2004.
    """
    def __init__(self, estimators):
        """
        Args:
            estimators(list)
        """
        self.estimators = estimators       
        
    def update_standalone_cv_predicts(self, nd_t1, nd_l1, nd_t2, update_list=None):     
        # Get predicts of different estimators
        for est in self.estimators:
            if (update_list is None) or (est in update_list):
                model = est.train(nd_t1, nd_l1)
                ypred = est.predict(model, nd_t2)
                dumpit(ypred, est.name+'_ypred')    
    
    def ensemble_select(self, nd_t1, nd_l1, nd_t2, nd_l2, loop, update_list=None):
        self.update_standalone_cv_predicts(nd_t1, nd_l1, nd_t2, update_list)
        ypreds = {}
        for est in self.estimators:
            ypreds[est.name] = loadit(est.name+'_ypred')
            #ypreds[est.name][ypreds[est.name]>3]=3
        # init weights
        record = []
        ensem_ypred = ypreds['xgb-9-3-0.8-1900']# + 0.5*ypreds['rfr-35-30']
        print('Initial ensemable score:', fmean_squared_error(nd_l2, ensem_ypred))
        w1, w2 = 0.99, 0.01
        # ensemble
        best_i, best_loop_score = 0, 1.
        for i in range(loop):
            best_score = 1.
            for name in ypreds.keys():   
                tmp_ypred = w1*ensem_ypred + w2*ypreds[name]
                score = fmean_squared_error(nd_l2, tmp_ypred)
                if score < best_score:
                    best_choice, best_w1, best_w2 = name, w1, w2
                    best_score = score
            ensem_ypred = best_w1*ensem_ypred + best_w2*ypreds[best_choice]
            record.append((best_choice, best_w1, best_w2))
            print('Ensemble score of round', i, ':', best_score, '; Choose', best_choice)
            if best_score < best_loop_score:
                best_loop_score = best_score
                best_i = i
        print('Best ensemble score:', best_loop_score)
        return record[:best_i+1]
            
    def update_standalone_predicts(self, nd_train, nd_label, nd_test, update_list=None):  
        for est in self.estimators:
            if (update_list is None) or (est in update_list):
                model = est.train(nd_train, nd_label)
                ypred = est.predict(model, nd_test)
                dumpit(ypred, est.name+'_ensem_ypred')
    
    def _get_ypred(self, name, nd_train, nd_label, nd_test):
        try:
            return loadit(name+'_ensem_ypred')
        except FileNotFoundError:
            for e in self.estimators:
                if e.name == name:
                    est = e
                    break
            self.update_standalone_predicts(nd_train, nd_label, nd_test, [est])
            return loadit(name+'_ensem_ypred')      
            
    def ensemble_predicts(self, record, nd_train, nd_label, nd_test, update_list=None):
        self.update_standalone_predicts(nd_train, nd_label, nd_test, update_list)
        name = 'xgb-9-3-0.8-1900'
        ensem_ypred = self._get_ypred(name, nd_train, nd_label, nd_test)# + 0.5*ypreds['rfr-35-30'] 
        for name, w1, w2 in record:
            ypred = self._get_ypred(name, nd_train, nd_label, nd_test)    
            ensem_ypred = w1*ensem_ypred + w2*ypred
        return ensem_ypred
            
            
            
