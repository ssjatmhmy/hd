import numpy as np
import pandas as pd
import config
from dataloader import DataLoader
from preprocessor import PreProcessor
from util import saveit, loadit, loadcsv, dumpit, submit
from feature_generator import FeatureGenerator, RFRFeatureGenerator, SECFeatureGenerator
from estimator import (RFREstimator, XGBEstimator, LassoEstimator, RidgeEstimator, 
                        KNNEstimator, GBDTEstimator, LSVREstimator)
import itertools
from sklearn import cross_validation
from hd_metrics import fmean_squared_error
from ensemble_selection import EnsembleSelection
from sklearn.decomposition import TruncatedSVD, NMF
import itertools
import matplotlib.pyplot as plt


class FeatureLoader(object):
    def __init__(self, DATA=False, NGRAM=False, W2VLEM=False, JOINNG=False):
        """
        Prepare the cleaned data, ngram, etc. for feature extraction.
        """
        # Create class instances.  
        self.dataloader = DataLoader(config) 
        self.preproc = PreProcessor(config)      
          
        if DATA is True:
            # Load the raw data and clean text
            df_data, self.n_train, self.nd_label = self.dataloader.load_and_merge_all()          
            df_data = self.preproc.clean_text(df_data)
            self.n_data = df_data.shape[0]
            saveit(df_data, 'df_data')
        else:
            self.n_train, self.nd_label = self.dataloader.get_n_train_and_label()
            if config.NUM_TEST is None:
                n_test = 166693
            else:
                n_test = config.NUM_TEST
            self.n_data = self.n_train + n_test
        # Get ngram columns.
        if NGRAM is True:
            df_ngram = self.preproc.get_ngram(loadit('df_data')) 
            saveit(df_ngram, 'df_ngram')
        # Join ngram
        if JOINNG is True:
            df_join_ng = self.preproc.join_ngram(loadit('df_ngram'))
            saveit(df_join_ng, 'df_join_ng')
        # Word2Vec Lemmatize
        if W2VLEM is True:
            df_data = loadit('df_data')
            df_w2vlem = self.preproc.w2vlem(df_data)
            saveit(df_w2vlem, 'df_w2vlem')
            df_w2vlem_join = self.preproc.join_w2vlem(df_data)
            saveit(df_w2vlem_join, 'df_w2vlem_join')

    def load(self):
        """
        Load all the features from last time's work.
        """
        df_feats = loadit('df_feats')      
        # Divide into train and test data
        nd_train = df_feats[:self.n_train].values
        nd_test = df_feats[self.n_train:].values
        return nd_train, nd_test, self.nd_label
        
    def re_load(self, **kargs):
        """
        Recompute features. 
        
        Set ALL to True if you want to recompute all the features.
        """
        WORDCOUNT, DIST, PUID, RFRIN, BRAND, TFIDF, TFIDFNMF = False,False,False,False,False,False,False
        COMPDIST, SEQ, COSDIST, W2V = False,False,False,False
        if 'ALL' in kargs.keys() and kargs['ALL'] is True:
            WORDCOUNT, DIST, PUID, RFRIN, BRAND, TFIDF, TFIDFNMF = True,True,True,True,True,True,True
            COMPDIST, SEQ, COSDIST, W2V = True,True,True,True
        for key in kargs.keys():
            if key == 'WORDCOUNT':
                WORDCOUNT = kargs['WORDCOUNT']
            if key == 'DIST':
                DIST = kargs['DIST']   
            if key == 'PUID':
                PUID = kargs['PUID']                            
            if key == 'RFRIN':
                RFRIN = kargs['RFRIN']
            if key == 'BRAND':
                BRAND = kargs['BRAND']   
            if key == 'TFIDF':
                TFIDF = kargs['TFIDF']
            if key == 'TFIDFNMF':
                TFIDFNMF = kargs['TFIDFNMF']
            if key == 'COMPDIST':
                COMPDIST = kargs['COMPDIST']            
            if key == 'SEQ':
                SEQ = kargs['SEQ']
            if key == 'COSDIST':
                COSDIST = kargs['COSDIST']                               
            if key == 'W2V':
                W2V = kargs['W2V']
                            
        # Feature extraction
        featgen = FeatureGenerator()
        # Get wordcount-related features
        if WORDCOUNT is True:
            featgen.extract_wordcount_related_feats(loadit('df_ngram'))  
        # Generate distance features
        if DIST is True:
            featgen.extract_distance_feats(loadit('df_ngram'))
        # Generate product uid feature
        if PUID is True:
            featgen.extract_puid_feats(loadit('df_data'))
            
        # RFR Feature extraction
        rfrfeatgen = RFRFeatureGenerator()
        # get rfr wordcount-related features
        if RFRIN is True:
            rfrfeatgen.extract_rfrin_feats(loadit('df_join_ng'))
        if BRAND is True:
            # get rfr brand feature
            rfrfeatgen.extract_brand_feats(loadit('df_ngram'), 
                                           loadit('df_data'), 
                                           loadit('df_wordcountfeats'))
        # get tfidf features
        if TFIDF is True:
            rfrfeatgen.extract_tfidf_feats(loadit('df_data'), 6)
        # get tfidf features
        if TFIDFNMF is True:
            rfrfeatgen.extract_tfidf_nmf_feats(loadit('df_data'), 20)
            
        # SEC Feature extraction
        secfeatgen = SECFeatureGenerator(config)      
        # get compression distance feature between 'search term' and 'title'   
        if COMPDIST is True:   
            secfeatgen.extract_compressdist_feat(loadit('df_data'))  
        # get SequenceMatcher related features besides compression distance
        if SEQ is True:   
            secfeatgen.extract_seq_related_feat(loadit('df_data')) 
        if COSDIST is True:
            secfeatgen.extract_cosinedist_feat(loadit('df_data'), loadit('df_w2vlem_join'))            
        if W2V is True:
            secfeatgen.extract_w2v_feat(loadit('df_w2vlem'))
                                    
        # Gather all features
        df_wordcountfeats = loadit('df_wordcountfeats')
        df_distancefeats = loadit('df_distancefeats')
        df_rfrin_feats = loadit('df_rfrin_feats')
        df_brand_feats = loadit('df_brand_feats')
        df_tfidf_feats = loadit('df_tfidf_feats') 
        df_compressdist_feats = loadit('df_compressdist_feats')
        df_seq_feats = loadit('df_seq_feats')    
        df_cosdist_feats = loadit('df_cosdist_feats') 
        df_w2v_feats = loadit('df_w2v_feats')      
        df_tsne_feats = loadcsv('df_tsne_feats')
         
        feat_list = [df_wordcountfeats, df_rfrin_feats, df_distancefeats,  \
                    df_brand_feats, df_tfidf_feats, df_compressdist_feats, df_seq_feats, \
                    df_cosdist_feats, df_w2v_feats, df_tsne_feats]      
                    
        df_feats = pd.DataFrame(index=range(self.n_data))
        for df in feat_list:
            df_feats = pd.merge(df_feats, df, left_index=True, right_index=True)
        saveit(df_feats, 'df_feats')
        
        # Divide into train and test data
        nd_train = df_feats[:self.n_train].values
        nd_test = df_feats[self.n_train:].values            
                   
        return nd_train, nd_test, self.nd_label

if __name__ == '__main__': 
    # Load features
    featloader = FeatureLoader(DATA=False, NGRAM=False, JOINNG=False, W2VLEM=False)
    nd_train, nd_test, nd_label = featloader.load()
    
    # Verify features. use KFold and xgb.
    if False: 
        kf = cross_validation.KFold(nd_train.shape[0], n_folds=4, shuffle=True, random_state=2016)
        kf_scores = []
        xgb_est = XGBEstimator() 
        for part1, part2 in kf:
            nd_t1, nd_l1 = nd_train[part1], nd_label[part1]
            nd_t2, nd_l2 = nd_train[part2], nd_label[part2]
            model = xgb_est.eval_train(nd_t1, nd_l1, nd_t2, nd_l2)
            ypred = xgb_est.eval_predict(model, nd_t2)
            score = fmean_squared_error(nd_l2, ypred)
            kf_scores.append(score)   
        print('Average kfold score of xgb:', sum(kf_scores)/len(kf_scores))
    
    # Ensemable selection.
    if True: 
        # Split train data into two part
        nd_t1, nd_t2, nd_l1, nd_l2 = cross_validation.train_test_split(\
                    nd_train, nd_label, test_size=0.65, random_state=2016)
        # prepare estimators   
        estimators = []
        ridge_estimators = []
        for alpha in np.arange(0.001,2,0.01):
            ridge_estimators.append(RidgeEstimator(alpha))
        estimators += ridge_estimators
        
        rfr_estimators = []
        for max_features in range(35,95,10): #range(55,95,10)
            for max_depth in range(10,40,10): #range(10,60,10)
                rfr_estimators.append(RFREstimator(max_features=max_features,max_depth=max_depth))
        estimators += rfr_estimators

        xgb_estimators = []
        for max_depth in range(3,13,2): 
            for min_child_weight in range(1,4,1): 
                 for colsample_bytree in np.arange(0.6,1.2,0.2): 
                    for n_estimators in range(1500,2100,200): 
                        xgb_estimators.append(XGBEstimator(max_depth, 
                                                           min_child_weight, 
                                                           colsample_bytree, 
                                                           n_estimators))
        estimators += xgb_estimators      
        
        ensem = EnsembleSelection(estimators)   
        # Get record
        record = ensem.ensemble_select(nd_t1, nd_l1, nd_t2, nd_l2, loop=500, update_list=ridge_estimators)
        # Ensemble predicts of different estimators
        ensem_ypred = ensem.ensemble_predicts(record, nd_train, nd_label, nd_test, update_list=[])
        
        submit(ensem_ypred)


