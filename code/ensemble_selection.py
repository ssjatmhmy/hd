from util import saveit, loadit, dumpit, submit
from estimator import (RFREstimator, XGBEstimator, LassoEstimator, RidgeEstimator, 
                        KNNEstimator, SVREstimator, KernelRidgeEstimator)
from hd_metrics import fmean_squared_error


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
        # init weights
        record = []
        ensem_ypred = 0.5*ypreds['xgboost'] + 0.5*ypreds['rfr-35-30']
        print(fmean_squared_error(nd_l2, ensem_ypred))
        self.w1, self.w2 = 0.8, 0.2
        # ensemble
        best_i, best_loop_score = 0, 1.
        for i in range(loop):
            best_score = 1.
            for name in ypreds.keys():   
                tmp_ypred = self.w1*ensem_ypred + self.w2*ypreds[name]
                score = fmean_squared_error(nd_l2, tmp_ypred)
                if score < best_score:
                    best_choice = name
                    best_score = score
            ensem_ypred = self.w1*ensem_ypred + self.w2*ypreds[best_choice]
            record.append(best_choice)
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
            
    def ensemble_predicts(self, record, nd_train, nd_label, nd_test, update_list=None):
        self.update_standalone_predicts(nd_train, nd_label, nd_test, update_list)
        ypreds = {}
        for est in self.estimators:
            ypreds[est.name] = loadit(est.name+'_ensem_ypred')
        ensem_ypred = 0.5*ypreds['xgboost'] + 0.5*ypreds['rfr-35-30']
        for name in record:
            ensem_ypred = self.w1*ensem_ypred + self.w2*ypreds[name]
        return ensem_ypred
            
            
            
