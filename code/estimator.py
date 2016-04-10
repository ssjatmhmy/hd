import hd_metrics
from sklearn import grid_search
from util import timethis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
import math
from sklearn import cross_validation
from abc import ABCMeta, abstractmethod


class BaseEstimator(object):
    def __init__(self, name):
        self.name = name
        

class RFREstimator(BaseEstimator):
    """
    skearn.ensemble.RandomForestRegressor
    """
    def __init__(self, max_features, max_depth):
        self.max_features = max_features
        self.max_depth = max_depth
        super(RFREstimator, self).__init__('-'.join(['rfr', str(max_features), str(max_depth)]))
    
    def cv(self, nd_train, nd_label):
        model = RandomForestRegressor(n_estimators=500, max_features=self.max_features, \
                    max_depth=self.max_depth, n_jobs=-1, random_state=2016, verbose=1)
        param_grid = {'max_features': [35], 'max_depth': [30]}
        model = grid_search.GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        return model    
        
    @timethis
    def train(self, nd_train, nd_label):
        print("Start training {:s}..".format(self.name))
        model = RandomForestRegressor(n_estimators=500, max_features=self.max_features, \
                    max_depth=self.max_depth, n_jobs=-1, random_state=2016, verbose=1)
        model.fit(nd_train, nd_label)
        return model
        
    def predict(self, model, nd_test):
        ypred = model.predict(nd_test)
        ypred[ypred<1]=1
        ypred[ypred>3]=3
        return ypred
                
        
class XGBEstimator(BaseEstimator):
    """
    xgboost
    """
    def __init__(self, max_depth, min_child_weight, colsample_bytree, n_estimators):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.n_estimators = n_estimators
        est_name = '-'.join(['xgb', 
                             str(max_depth), 
                             str(min_child_weight), 
                             str(colsample_bytree),
                             str(n_estimators)])
        super(XGBEstimator, self).__init__(est_name)
        
    @timethis
    def xgb_cv(self, nd_train, nd_label):
        dtrain = xgb.DMatrix(nd_train, label=nd_label)
        param = {'bst:max_depth':10, 'bst:eta':0.3, 'silent':1, 'objective':'reg:linear' }
        param['nthread'] = 4
        param['eval_metric'] = 'rmse'
        num_round = 30
        hist = xgb.cv(param, dtrain, num_round, nfold=2)
        print(hist)
    
    def plot_importance(self, model):
        xgb.plot_importance(model)
    
    @timethis
    def cv(self, nd_train, nd_label):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(\
                        nd_train, nd_label, test_size=0.5, random_state=0)
        bst = self.train(X_train, y_train)
        ypred = self.predict(bst, X_test)
        print('cv score:', hd_metrics.fmean_squared_error(y_test, ypred))
    
    @timethis
    def train(self, nd_train, nd_label):
        print("Start training {:s}..".format(self.name))
        rgr = xgb.XGBRegressor(max_depth = self.max_depth,
                        learning_rate = 0.01,
                        silent = 1,
                        objective = 'reg:linear',
                        nthread = 4,
                        min_child_weight = self.min_child_weight,
                        colsample_bytree = self.colsample_bytree,
                        subsample = .85,
                        n_estimators = self.n_estimators)
        bst = rgr.fit(nd_train, nd_label, eval_metric='rmse')
        return bst
 
    @timethis
    def eval_train(self, nd_train, nd_label, nd_eval, nd_evallabel):
        """
        Best param:
            max_depth = 9,
            learning_rate = 0.01,
            silent = 1,
            objective = 'reg:linear',
            nthread = 4,
            min_child_weight = 3,
            colsample_bytree = .8,
            subsample = .85,
            n_estimators=2000
        """
        print("Start training {:s}..".format(self.name))
        rgr = xgb.XGBRegressor(max_depth = self.max_depth,
                        learning_rate = 0.01,
                        silent = 1,
                        objective = 'reg:linear',
                        nthread = 4,
                        min_child_weight = self.min_child_weight,
                        colsample_bytree = self.colsample_bytree,
                        subsample = .85,
                        n_estimators = self.n_estimators) 
        bst = rgr.fit(nd_train, nd_label, early_stopping_rounds=10, eval_metric='rmse',
                        eval_set=[(nd_eval, nd_evallabel)])
        return bst 
    
    def eval_predict(self, bst, nd_test):
        ypred = bst.predict(nd_test, ntree_limit=bst.best_ntree_limit)
        ypred[ypred<1]=1
        ypred[ypred>3]=3
        return ypred    
    
    def predict(self, bst, nd_test):
        ypred = bst.predict(nd_test)
        ypred[ypred<1]=1
        ypred[ypred>3]=3
        return ypred
        

class GBDTEstimator(BaseEstimator):
    """
    skearn.ensemble.GradientBoostingRegressor
    To slow (>3h). Consider dropping this one.
    """
    def __init__(self, max_features, max_depth):
        self.max_features = max_features
        self.max_depth = max_depth
        super(GBDTEstimator, self).__init__('-'.join(['gbdt', str(max_features), str(max_depth)]))
    
    def cv(self, nd_train, nd_label):
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=2016)
        param_grid = {'max_features': [20,35], 'max_depth': [30]}
        model = grid_search.GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        return model    
        
    @timethis
    def train(self, nd_train, nd_label):
        print("Start training {:s}..".format(self.name))
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, \
                    max_features=self.max_features, \
                    max_depth=self.max_depth, random_state=2016)
        model.fit(nd_train, nd_label)
        return model
        
    def predict(self, model, nd_test):
        ypred = model.predict(nd_test)
        ypred[ypred<1]=1
        ypred[ypred>3]=3
        return ypred


class LassoEstimator(BaseEstimator):
    """
    skearn.linear_model.Lasso
    """
    def __init__(self):
        super(LassoEstimator, self).__init__('lasso')
        
    @timethis
    def train(self, nd_train, nd_label):
        print("Start training {:s}..".format(self.name))
        lso = linear_model.Lasso(alpha = 0.1)
        param_grid = {'alpha': [0.001], 'max_iter': [4000]}
        model = grid_search.GridSearchCV(estimator=lso, param_grid=param_grid, n_jobs=-1, cv=2, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        return model
        
    def predict(self, model, nd_test):
        ypred = model.predict(nd_test)
        ypred[ypred<1]=1
        ypred[ypred>3]=3
        return ypred
        
        
class RidgeEstimator(BaseEstimator):
    """
    skearn.linear_model.Ridge
    """
    def __init__(self, alpha):
        self.alpha = alpha
        est_name = '-'.join(['ridge', str(alpha)])
        super(RidgeEstimator, self).__init__(est_name)
        
    @timethis
    def cv(self, nd_train, nd_label):
        print("Start training {:s}..".format(self.name))
        rid = linear_model.Ridge(alpha = 0.5, max_iter = 4000)
        param_grid = {'alpha': [0.5], 'max_iter': [4000]}
        model = grid_search.GridSearchCV(estimator=rid, param_grid=param_grid, n_jobs=-1, cv=2, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        return model
        
    @timethis
    def train(self, nd_train, nd_label):
        print("Start training {:s}..".format(self.name))
        model = linear_model.Ridge(alpha = self.alpha, max_iter = 4000)
        model.fit(nd_train, nd_label)
        return model 
        
    def predict(self, model, nd_test):
        ypred = model.predict(nd_test)
        ypred[ypred<1]=1
        ypred[ypred>3]=3
        return ypred
        
        
class KNNEstimator(BaseEstimator):
    """
    sklearn.neighbors.KNeighborsRegressor
    """
    def __init__(self):
        super(KNNEstimator, self).__init__('knn')
        
    @timethis
    def train(self, nd_train, nd_label):
        print("Start training {:s}..".format(self.name))
        neigh = KNeighborsRegressor()
        param_grid = {'n_neighbors': [80], 'p':[1]}
        model = grid_search.GridSearchCV(estimator=neigh, param_grid=param_grid, n_jobs=-1, cv=2, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        return model
        
    def predict(self, model, nd_test):
        ypred = model.predict(nd_test)
        ypred[ypred<1]=1
        ypred[ypred>3]=3
        return ypred
        
        
class KernelRidgeEstimator(BaseEstimator):
    """
    skearn.kernel_ridge.KernelRidge
    MemoryError. Consider dropping this one.
    """
    @timethis
    def train(self, nd_train, nd_label):
        rid = KernelRidge()
        #param_grid = {'kernel': ['linear','rbf','poly','sigmoid'], 'alpha': [0.1,0.5,1.0]}
        param_grid = {'kernel': ['rbf'], 'alpha': [1.0]}
        model = grid_search.GridSearchCV(estimator=rid, param_grid=param_grid, n_jobs=-1, cv=3, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        
        
class ElasticNetEstimator(BaseEstimator):
    """
    skearn.linear_model.ElasticNet
    Consider dropping this one because objective does not converge.
    """
    @timethis
    def train(self, nd_train, nd_label):
        clf = linear_model.ElasticNet(alpha = 0.1)
        param_grid = {'alpha': [0.01], 'l1_ratio': [0.01], 'max_iter': [4000]}
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)        
        
        
class LSVREstimator(BaseEstimator):
    """
    skearn.svm.SVR
    Cost too much time. Consider dropping this one.
    """
    def __init__(self):
        super(LSVREstimator, self).__init__('lsvr')
    
    @timethis
    def cv(self, nd_train, nd_label):
        model = svm.SVR()#LinearSVR(random_state=2016)
        param_grid = {'C':[0.04]}
        model = grid_search.GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)   
        return model

    @timethis
    def train(self, nd_train, nd_label):
        print("Start training {:s}..".format(self.name))
        model = svm.LinearSVR(dual=True,random_state=2016)
        model.fit(nd_train, nd_label)
        return model
        
    def predict(self, model, nd_test):
        ypred = model.predict(nd_test)
        ypred[ypred<1]=1
        ypred[ypred>3]=3
        return ypred    

        
        
        
