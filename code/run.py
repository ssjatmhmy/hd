import numpy as np
import pandas as pd
import config
from dataloader import DataLoader
from preprocessor import PreProcessor
from util import saveit, loadit, dumpit, submit
from feature_generator import FeatureGenerator, RFRFeatureGenerator, SECFeatureGenerator
from estimator import (RFREstimator, XGBEstimator, LassoEstimator, RidgeEstimator, 
                        ElasticNetEstimator, SVREstimator, KernelRidgeEstimator)
import itertools
from sklearn import cross_validation
from hd_metrics import fmean_squared_error


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
        WORDCOUNT, DIST, RFRIN, BRAND, TFIDF = False,False,False,False,False
        COMPDIST, SEQ, COSDIST, W2V = False,False,False,False
        if 'ALL' in kargs.keys() and kargs['ALL'] is True:
            WORDCOUNT, DIST, RFRIN, BRAND, TFIDF = True,True,True,True,True
            COMPDIST, SEQ, COSDIST, W2V = True,True,True,True
        for key in kargs.keys():
            if key == 'WORDCOUNT':
                WORDCOUNT = kargs['WORDCOUNT']
            if key == 'DIST':
                DIST = kargs['DIST']            
            if key == 'RFRIN':
                RFRIN = kargs['RFRIN']
            if key == 'BRAND':
                BRAND = kargs['BRAND']   
            if key == 'TFIDF':
                TFIDF = kargs['TFIDF']
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
            
        # RFR Feature extraction
        rfrfeatgen = RFRFeatureGenerator()
        # get rfr wordcount-related features
        if RFRIN is True:
            rfrfeatgen.extract_rfrin_feats(loadit('df_join_ng'))
        if BRAND is True:
            # get rfr brand feature
            rfrfeatgen.extract_brand_feats(loadit('df_ngram'), loadit('df_data'), loadit('df_wordcountfeats'))
        # get tfidf features
        if TFIDF is True:
            rfrfeatgen.extract_tfidf_feats(loadit('df_data'), 7)
            
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
        
        feat_list = [df_wordcountfeats, df_rfrin_feats, df_distancefeats, df_brand_feats, \
                    df_tfidf_feats, df_compressdist_feats, df_seq_feats, df_cosdist_feats, \
                    df_w2v_feats]
                    
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
    
    # xgboost estimator
    #est = XGBEstimator()
    #xgbest.cv(nd_train, nd_label)
    #model = est.train(nd_train, nd_label)
    #xg_ypred = est.predict(model, nd_test)

    # Split train data into two part
    nd_t1, nd_t2, nd_l1, nd_l2 = cross_validation.train_test_split(\
                nd_train, nd_label, test_size=0.5, random_state=2016)
                
    if False:    
        # Get predicts of different estimators
        ypreds = {}     
        for est in [XGBEstimator(), LassoEstimator(), RidgeEstimator(), RFREstimator()]:
            model = est.train(nd_t1, nd_l1)
            ypred = est.predict(model, nd_t2)
            ypreds[est.name] = ypred
        dumpit(ypreds, 'ypreds')    
    
    ypreds = loadit('ypreds')
    # init weights
    record = []
    ensem_ypred = ypreds['xgboost']# + 0.5*ypreds['rfr']
    print(fmean_squared_error(nd_l2, ensem_ypred))
    w1, w2 = 0.5, 0.5
    # ensemble
    for i in range(10):
        best_score = 1.
        for name in ypreds.keys():   
            tmp_ypred = w1*ensem_ypred + w2*ypreds[name]
            score = fmean_squared_error(nd_l2, tmp_ypred)
            if score < best_score:
                best_choice = name
                best_score = score
        ensem_ypred = w1*ensem_ypred + w2*ypreds[best_choice]
        record.append(best_choice)
        print('Ensemble score of round', i, ':', best_score)
    print(record)
    
    # Ensemble predicts of different estimators
    if True:
        ypreds = {}     
        for est in [XGBEstimator(), LassoEstimator(), RidgeEstimator(), RFREstimator()]:
            model = est.train(nd_train, nd_label)
            ypred = est.predict(model, nd_test)
            ypreds[est.name] = ypred
            
        ensem_ypred = 0.5*ypreds['xgboost'] + 0.5*ypreds['rfr']
        #for name in record:
        #    ensem_ypred = w1*ensem_ypred + w2*ypreds[name]
        submit(ensem_ypred)


