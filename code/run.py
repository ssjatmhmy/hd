import numpy as np
import pandas as pd
import config
from dataloader import DataLoader
from preprocessor import PreProcessor
from util import saveit, loadit
from feature_generator import FeatureGenerator, RFRFeatureGenerator, SECFeatureGenerator
from estimator import (RFREstimator, XGBEstimator, LassoEstimator, RidgeEstimator, 
                        ElasticNetEstimator, SVREstimator, KernelRidgeEstimator)


class FeatureLoader(object):
    def __init__(self, DATA=False, NGRAM=False, W2VLEM=False):
        """
        Prepare the cleaned data, ngram, etc. for feature extraction.
        """
        # Create class instances.  
        self.dataloader = DataLoader(config) 
        self.preproc = PreProcessor(config)      
          
        if DATA is True:
            # Load the raw data and clean text
            df_data, self.n_train, self.nd_label = self.dataloader.load_and_merge_all()            
            self.df_data = self.preproc.clean_text(df_data)
            saveit(self.df_data, 'df_data')
        else:
            # Load cleanned data
            self.df_data = loadit('df_data')
            self.n_train, self.nd_label = self.dataloader.get_n_train_and_label()
        # Get ngram columns.
        if NGRAM is True:
            self.df_ngram = self.preproc.get_ngram(self.df_data) 
            saveit(self.df_ngram, 'df_ngram')
        else:
            self.df_ngram = loadit('df_ngram')
        # Join ngram
        self.df_join_ng = self.preproc.join_ngram(self.df_ngram)
        # Word2Vec Lemmatize
        if W2VLEM is True:
            self.df_w2vlem = self.preproc.w2vlem(self.df_data)
            saveit(self.df_w2vlem, 'df_w2vlem')
            self.df_w2vlem_join = self.preproc.join_w2vlem(self.df_data)
            saveit(self.df_w2vlem_join, 'df_w2vlem_join')
        else:
            self.df_w2vlem = loadit('df_w2vlem')
            self.df_w2vlem_join = loadit('df_w2vlem_join')

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
            featgen.extract_wordcount_related_feats(self.df_ngram)  
        # Generate distance features
        if DIST is True:
            featgen.extract_distance_feats(self.df_ngram)
            
        # RFR Feature extraction
        rfrfeatgen = RFRFeatureGenerator()
        # get rfr wordcount-related features
        if RFRIN is True:
            rfrfeatgen.extract_rfrin_feats(self.df_join_ng)
        if BRAND is True:
            # get rfr brand feature
            rfrfeatgen.extract_brand_feats(self.df_ngram, self.df_data, loadit('df_wordcountfeats'))
        # get tfidf features
        if TFIDF is True:
            rfrfeatgen.extract_tfidf_feats(self.df_data, 10)
            
        # SEC Feature extraction
        secfeatgen = SECFeatureGenerator(config)      
        # get compression distance feature between 'search term' and 'title'   
        if COMPDIST is True:   
            secfeatgen.extract_compressdist_feat(self.df_data)  
        # get SequenceMatcher related features besides compression distance
        if SEQ is True:   
            secfeatgen.extract_seq_related_feat(self.df_data) 
        if COSDIST is True:
            secfeatgen.extract_cosinedist_feat(self.df_data, self.df_join_ng)            
        if W2V is True:
            secfeatgen.extract_w2v_feat(self.df_w2vlem)
                                    
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
        
        df_feats = pd.DataFrame(index=range(self.df_data.shape[0]))
        feat_list = [df_wordcountfeats, df_rfrin_feats, df_distancefeats, df_brand_feats, \
                    df_tfidf_feats, df_compressdist_feats, df_seq_feats, df_cosdist_feats, \
                    df_w2v_feats]
        for df in feat_list:
            df_feats = pd.merge(df_feats, df, left_index=True, right_index=True)
        saveit(df_feats, 'df_feats')

        # Divide into train and test data
        nd_train = df_feats[:self.n_train].values
        nd_test = df_feats[self.n_train:].values

        return nd_train, nd_test, self.nd_label

if __name__ == '__main__': 
    # Load features
    featloader = FeatureLoader(DATA=True, NGRAM=True, W2VLEM=True)
    nd_train, nd_test, nd_label = featloader.re_load(ALL=True)

    # xgboost estimator
    XGBEstimator().train(nd_train, nd_label)

    # Lasso estimator
    #LassoEstimator().train(nd_train, nd_label)
    
    # Ridge estimator
    #RidgeEstimator().train(nd_train, nd_label)
    
    # RF estimator
    #RFREstimator().train(nd_train, nd_label)



