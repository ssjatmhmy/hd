from tsne import bh_sne
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import cPickle


def extract_tsne_feat():
    """
    Extract tsne features.
    Note: python2 only.    
    """  
    df_w2vlem_join = pd.read_csv('tmp2/df_w2vlem_join.csv', index_col=0)
         
    df_feat = pd.DataFrame(index=df_w2vlem_join.index.values)
    tfidf = TfidfVectorizer(ngram_range=(1,4), stop_words='english', min_df=2) 
    X_t = tfidf.fit_transform(df_w2vlem_join['t_w2v'].tolist())    
     
    svd = TruncatedSVD(n_components=100, random_state=2016)     
    X_svd = svd.fit_transform(X_t)
    X_scaled = StandardScaler().fit_transform(X_svd)
    X_tsne = bh_sne(X_scaled)
    df_feat['tsne_t_1'] = X_tsne[:len(df_w2vlem_join), 0]
    df_feat['tsne_t_2'] = X_tsne[:len(df_w2vlem_join), 1]
    df_feat.to_csv('tmp2/tsne_t', index=False)

    print(df_feat)
    tfidf = TfidfVectorizer(ngram_range=(1,4), stop_words='english', min_df=2) 
    X_q = tfidf.fit_transform(df_w2vlem_join['q_w2v'].tolist())
    X_tq = sp.hstack([X_t, X_q]).tocsr()
    svd = TruncatedSVD(n_components=100, random_state=2016)
    X_svd = svd.fit_transform(X_tq)
    X_scaled = StandardScaler().fit_transform(X_svd)
    X_tsne = bh_sne(X_scaled)
    df_feat['tsne_qt_1'] = X_tsne[:len(df_w2vlem_join), 0]
    df_feat['tsne_qt_2'] = X_tsne[:len(df_w2vlem_join), 1]
    df_feat.to_csv('tmp2/tsne_qt', index=False)

    df_feat = pd.read_csv('tmp2/tsne_qt')
    print(df_feat)    
    tfidf = TfidfVectorizer(ngram_range=(1,3), stop_words='english', min_df=2) 
    X_d = tfidf.fit_transform(df_w2vlem_join['d_w2v'].tolist())
    svd = TruncatedSVD(n_components=70, random_state=2016)
    X_svd = svd.fit_transform(X_d)
    X_scaled = StandardScaler().fit_transform(X_svd)
    X_tsne = bh_sne(X_scaled)
    df_feat['tsne_desc_1'] = X_tsne[:len(df_w2vlem_join), 0]
    df_feat['tsne_desc_2'] = X_tsne[:len(df_w2vlem_join), 1]
    
    df_tsne_feats = df_feat
    df_tsne_feats.to_csv('tmp2/df_tsne_feats.csv')
    

def extract_tsne_gather_feat(stage):
    """
    Extract tsne gather features.
    Note: python2 only.    
    Better than func:extract_tsne_feat in cv, but worst in submission.
    """  
    df_w2vlem_join = pd.read_csv('tmp2/df_w2vlem_join.csv', index_col=0)
        
    if stage <= 1:        
        df_feat = pd.DataFrame(index=df_w2vlem_join.index.values)
        tfidf = TfidfVectorizer(ngram_range=(2,4), stop_words='english', min_df=2)
        
        df_w2vlem_join['t_w2v'].to_csv('tmp2/t_w2v', index=False)
        df_w2vlem_join['q_w2v'].to_csv('tmp2/q_w2v', index=False)
        df_w2vlem_join['d_w2v'].to_csv('tmp2/d_w2v', index=False)
        
        tfidf.set_params(input='filename')        
        tfidf.fit(['tmp2/t_w2v','tmp2/q_w2v','tmp2/d_w2v'])
        tfidf.set_params(input='content')
        
        cPickle.dump(tfidf, open('tmp2/tfidf_obj','wb'))
    
    tfidf = cPickle.load(open('tmp2/tfidf_obj','rb'))
    X_t = tfidf.transform(df_w2vlem_join['t_w2v'].tolist())    
    if stage <= 2:           
        svd = TruncatedSVD(n_components=100, random_state=2016)     
        X_svd = svd.fit_transform(X_t)
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = bh_sne(X_scaled)
        df_feat['tsne_t_1'] = X_tsne[:len(df_w2vlem_join), 0]
        df_feat['tsne_t_2'] = X_tsne[:len(df_w2vlem_join), 1]
        df_feat.to_csv('tmp2/tsne_t', index=False)
    
    df_feat = pd.read_csv('tmp2/tsne_t')    
    if stage <= 3:
        print(df_feat)
        X_q = tfidf.transform(df_w2vlem_join['q_w2v'].tolist())
        X_tq = sp.hstack([X_t, X_q]).tocsr()
        svd = TruncatedSVD(n_components=50, random_state=2016)
        X_svd = svd.fit_transform(X_tq)
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = bh_sne(X_scaled)
        df_feat['tsne_qt_1'] = X_tsne[:len(df_w2vlem_join), 0]
        df_feat['tsne_qt_2'] = X_tsne[:len(df_w2vlem_join), 1]
        df_feat.to_csv('tmp2/tsne_qt', index=False)
    
    df_feat = pd.read_csv('tmp2/tsne_qt')    
    if stage <= 4:
        print(df_feat)    
        X_d = tfidf.transform(df_w2vlem_join['d_w2v'].tolist())
        svd = TruncatedSVD(n_components=100, random_state=2016)
        X_svd = svd.fit_transform(X_d)
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = bh_sne(X_scaled)
        df_feat['tsne_desc_1'] = X_tsne[:len(df_w2vlem_join), 0]
        df_feat['tsne_desc_2'] = X_tsne[:len(df_w2vlem_join), 1]
        
        df_tsne_feats = df_feat
        df_tsne_feats.to_csv('tmp2/df_tsne_gather_feats.csv')

def extract_w2v_tsne_feat():
    """
    Extract w2v tsne features.
    Note: python2 only. Worst in cv, so do not use this.   
    """  
    df_w2v_feats = pd.read_csv('tmp2/df_w2v_feats.csv', index_col=0)
    X = df_w2v_feats.values
         
    df_feat = pd.DataFrame(index=df_w2v_feats.index.values)
    
    X_scaled = StandardScaler().fit_transform(X)
    X_tsne = bh_sne(X_scaled)
    df_feat['tsne_t_1'] = X_tsne[:len(df_w2v_feats), 0]
    df_feat['tsne_t_2'] = X_tsne[:len(df_w2v_feats), 1]
    df_feat.to_csv('tmp2/df_tsne_w2v_feats.csv')
    
if __name__ == '__main__': 
    extract_tsne_feat()

