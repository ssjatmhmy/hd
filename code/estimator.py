import hd_metrics
from sklearn import grid_search
from util import timethis
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import math
from sklearn import cross_validation


class BaseEstimator(object):
    def __init__(self):
        print("Start training..")


class RFREstimator(BaseEstimator):
    """
    skearn.ensemble.RandomForestRegressor
    """
    @timethis
    def train(self, nd_train, nd_label):
        rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=2016, verbose=1)
        param_grid = {'max_features': [35], 'max_depth': [30]}
        model = grid_search.GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=-1, cv=2, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        
        
class XGBEstimator(BaseEstimator):
    """
    xgboost
    """
    @timethis
    def xgb_cv(self, nd_train, nd_label):
        dtrain = xgb.DMatrix(nd_train, label=nd_label)
        param = {'bst:max_depth':10, 'bst:eta':0.3, 'silent':1, 'objective':'reg:linear' }
        param['nthread'] = 4
        param['eval_metric'] = 'rmse'
        num_round = 30
        hist = xgb.cv(param, dtrain, num_round, nfold=2)
        print(hist)

    def cv(self, nd_train, nd_label):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(\
                        nd_train, nd_label, test_size=0.5, random_state=0)
        bst = self.train(X_train, y_train)
        ypred = self.predict(bst, X_test)
        print('cv score:', hd_metrics.fmean_squared_error(y_test, ypred))
    
    def train(self, nd_train, nd_label):
        dtrain = xgb.DMatrix(nd_train, label=nd_label)
        param = {'bst:max_depth':10, 'bst:eta':0.3, 'silent':1, 'objective':'reg:linear' }
        param['nthread'] = 4
        param['eval_metric'] = 'rmse'
        plst = param.items()
        num_round = 30    
        bst = xgb.train(plst, dtrain, num_round)
        return bst
        
    def predict(self, bst, nd_test):
        dtest = xgb.DMatrix(nd_test)
        ypred = bst.predict(dtest)
        return ypred
        

class LassoEstimator(BaseEstimator):
    """
    skearn.linear_model.Lasso
    """
    @timethis
    def train(self, nd_train, nd_label):
        lso = linear_model.Lasso(alpha = 0.1)
        param_grid = {'alpha': [0.001], 'max_iter': [4000]}
        model = grid_search.GridSearchCV(estimator=lso, param_grid=param_grid, n_jobs=-1, cv=2, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        
        
class RidgeEstimator(BaseEstimator):
    """
    skearn.linear_model.Ridge
    """
    @timethis
    def train(self, nd_train, nd_label):
        rid = linear_model.Ridge(alpha = 0.1)
        param_grid = {'alpha': [0.5], 'max_iter': [4000]}
        model = grid_search.GridSearchCV(estimator=rid, param_grid=param_grid, n_jobs=-1, cv=2, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        
        
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
        
        
class SVREstimator(BaseEstimator):
    """
    skearn.svm.SVR
    Cost too much time. Consider dropping this one.
    """
    @timethis
    def train(self, nd_train, nd_label):
        clf = SVR()
        #param_grid = {'kernel': ['linear','rbf','poly','sigmoid'], 'C': [0.1, 1.0, 2.0]}
        param_grid = {'kernel': ['linear'], 'C': [1.0]}
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, \
                                                   verbose = 1, scoring=hd_metrics.RMSE)
        model.fit(nd_train, nd_label)
        
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)   
        
        
        
        
