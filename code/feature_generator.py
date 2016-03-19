import ipdb
import pandas as pd
import numpy as np
import re
import itertools
from util import timethis
from util import saveit, loadit
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import lzma
from difflib import SequenceMatcher
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler


def stem_words(words, stemmer):
    for w in words.split():
        yield stemmer.stem(w) 


def stem_text(sr, stem_type):
    """
    Stem text of a pd.Series.
    
    Args:
        sr(pd.Series): To be stemmed.
        stem_type(str): Choose 'porter' or 'snowball'.
        
    Returns:
        sr_stemmed(pd.Series)
    """
    # Load stemmer
    if stem_type == 'porter':
        stemmer = nltk.stem.PorterStemmer()
    elif stem_type == "snowball":
        stemmer = nltk.stem.SnowballStemmer('english')
    # Stemming
    #stem_words = lambda s: yield stemmer.stem(w) for w in s.split()
    stem_func = lambda s: " ".join(stem_words(s, stemmer))
    sr_stemmed = sr.map(lambda x:stem_func(str(x)))     
    return sr_stemmed
    
    
class FeatureGenerator(object):
    """
    Generate ad hoc features.
    """
    def __init__(self):
        pass

    def _gen_inter_count_rate_feats(self, df_data, col1, col2):
        """
        Generate intersect word count/rate features.
        Note: inplace operation on ivar:df_data.
        """
        # Generate intersect word count features
        count_func = lambda x: sum([1. for w in x[col1] if w in x[col2]])
        countcol = '_'.join([col1,col2,'count'])
        df_data[countcol] = df_data.apply(count_func, axis=1)
        # Generate intersect word rate features
        ratecol = '_'.join([col1,col2,'rate'])
        df_data[ratecol] = df_data[countcol].div(df_data[col1+'_listlen']).fillna(0)
                    
    def _gen_inter_query_others_feats(self, df_data, col1, col2):
        """
        Generate other intersect features (between query (i.e., "search_term") and others).
        Note: inplace operation on ivar:df_data.
        """
        countcol = '_'.join([col1,col2,'count'])
        ratecol = '_'.join([col1,col2,'rate_addon1'])
        df_data[ratecol] = df_data[countcol].div(df_data[col2+'_listlen']).fillna(0)
        ratecol = '_'.join([col1,col2,'rate_addon2'])
        div_countcol = '_'.join([col2,col1,'count'])
        df_data[ratecol] = df_data[countcol].div(df_data[div_countcol]).fillna(0)        
                         
    def _get_position_list(self, obs, target):
        """
        Get the list of positions of obs in target
        """
        if len(obs) == 0: return [0]
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0: return [0]
        return pos_of_obs_in_target
                                   
    def _gen_pos_feats(self, df_data, col1, col2):
        """
        Generate intersect word position features.
        Note: inplace operation on ivar:df_data.
        """
        pos_func = lambda x: self._get_position_list(x[col1], x[col2])                           
        sr = df_data.apply(pos_func, axis=1)
        # Statistic features on position
        col = '_'.join([col1,col2,'pos'])  
        df_data[col+'_min'] = sr.map(np.min)
        df_data[col+'_mean'] = sr.map(np.mean)
        df_data[col+'_median'] = sr.map(np.median)
        df_data[col+'_max'] = sr.map(np.max)
        df_data[col+'_std'] = sr.map(np.std)
        # Statistic features on normalized position
        df_data[col+'_min_nml'] = df_data[col+'_min'].div(df_data[col1+'_listlen']).fillna(0)
        df_data[col+'_mean_nml'] = df_data[col+'_mean'].div(df_data[col1+'_listlen']).fillna(0)
        df_data[col+'_median_nml'] = df_data[col+'_median'].div(df_data[col1+'_listlen']).fillna(0)
        df_data[col+'_max_nml'] = df_data[col+'_max'].div(df_data[col1+'_listlen']).fillna(0)
        df_data[col+'_std_nml'] = df_data[col+'_std'].div(df_data[col1+'_listlen']).fillna(0)
        
    @timethis
    def extract_wordcount_related_feats(self, df_ngram):
        """
        Extract word count related features corresponding to ngram. 
        
        Args:
            df_ngram(pd.DataFrame): The data dataframe that contains ngram columns; could be train \
                                        data, test data or both.
        """
        df_feat = pd.DataFrame()
        # Generate word count features corresponding to ngram. 
        # Only process the columns that can be used to generate word count features.
        for ngcol in df_ngram.columns:
            df_feat[ngcol+'_listlen'] = df_ngram[ngcol].map(len)
            df_feat[ngcol+'_setlen'] = df_ngram[ngcol].map(lambda x: len(set(x)))
            df_feat[ngcol+'_setlen/listlen'] = df_feat[ngcol+'_setlen'].div\
                                                (df_feat[ngcol+'_listlen']).fillna(0)
            # Generate unigram digit count features.
            if ngcol.endswith('unigram'):
                count_digit_func = lambda x: sum([1. for w in x if w.isdigit()])
                df_feat[ngcol+'_digitcount'] = df_ngram[ngcol].map(count_digit_func)
                df_feat[ngcol+'_digitrate'] = df_feat[ngcol+'_digitcount'].div\
                                                (df_feat[ngcol+'_listlen']).fillna(0)
        
        # Generate intersect word/digit count/rate/position features corresponding to ngram. 
        df_merge = pd.merge(df_feat, df_ngram, left_index=True, right_index=True)
        suffixs = ['_unigram', '_bigram'] # better exclude '_trigram'
        for suffix in suffixs:   
            # Intersect of each pair  
            inter_columns = ["q", "t", "d"]
            for col1, col2 in itertools.permutations(inter_columns,2):   
                col1 += suffix
                col2 += suffix
                self._gen_inter_count_rate_feats(df_merge, col1, col2)
                self._gen_pos_feats(df_merge, col1, col2)
            # Intersect between "search_term" (i.e., "q") and the others (i.e., "t" and "d")
            for col1 in ["t", "d"]:
                col1 += suffix
                col2 = "q"+suffix
                self._gen_inter_query_others_feats(df_merge, col1, col2) 
            # Intersect between "search_term" (i.e., "q") and "brand" (i.e., "b")
            col1 = "q"+suffix
            col2 = "b"+suffix
            self._gen_inter_count_rate_feats(df_merge, col1, col2)
            
        df_wordcountfeats = df_merge.drop(df_ngram.columns, axis=1)
        saveit(df_wordcountfeats, 'df_wordcountfeats')
                                        
    def _jaccard_coef(self, list1, list2):
        """
        Compute Jaccard coefficient between list1 and list2.
        """
        set1, set2 = set(list1), set(list2)
        intersect = len(set1.intersection(set2))
        union = len(set1.union(set2))
        try:
            return intersect/union
        except ZeroDivisionError:
            return 0.0

    def _dice_dist(self, list1, list2):
        """
        Compute dice distance between list1 and list2.
        """
        set1, set2 = set(list1), set(list2)
        intersect = len(set1.intersection(set2))
        union = len(set1) + len(set2)
        try:
            return 2.0*intersect/union
        except ZeroDivisionError:
            return 0.0        

    @timethis
    def extract_distance_feats(self, df_ngram):    
        """
        Extract Jaccard coefficient and dice distance features.
        
        Args:
            df_ngram(pd.DataFrame): The dataframe that contains ngram columns; could be train \
                                        data, test data or both.
        Returns:
            df_feat(pd.DataFrame)
        """
        df_feat = pd.DataFrame()
        suffixs = ['_bigram']#'_unigram', , '_trigram'
        inter_columns = ["q", "t"] #, "d" better exclude 'd' and 'b'
        for suffix in suffixs:   
            for col1, col2 in itertools.combinations(inter_columns,2):
                col1 += suffix  
                col2 += suffix  
                # Jaccard coefficient features   
                jacc_func = lambda x: self._jaccard_coef(x[col1], x[col2])
                df_feat['_'.join([col1,col2,'jacc'])] = df_ngram.apply(jacc_func, axis=1)   
                # Dice distance features        
                dice_func = lambda x: self._dice_dist(x[col1], x[col2])
                df_feat['_'.join([col1,col2,'dice'])] = df_ngram.apply(dice_func, axis=1)   
        df_distancefeats =  df_feat
        saveit(df_distancefeats, 'df_distancefeats')
            
            
class RFRFeatureGenerator(object):
    """
    Generate RFR features.
    Reference the public script 'RFR_Features.py'
    """
    def __init__(self):
        pass
        
    def _seg_words(self, list1, list2):
        tmplist2 = set()
        for z in list2:
            z = re.sub("[^a-z0-9./]", '', z)
            if len(z)>2:
                tmplist2.add(z)
        tmplist2 = list(tmplist2)
        s = []
        for word in list1:
            if len(word)>3:
                s1 = []
                s1 += self._segmentit(word,tmplist2,True)
                if len(s)>1:
                    s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
                else:
                    s.append(word)
            else:
                s.append(word)
        return (" ".join(s))

    def _segmentit(self, s, txt_arr, t):
        st = s
        r = []
        for j in range(len(s)):
            for word in txt_arr:
                if word == s[:-j]:
                    r.append(s[:-j])
                    #print(s[:-j],s[len(s)-j:])
                    s=s[len(s)-j:]
                    r += self._segmentit(s, txt_arr, False)
        if t:
            i = len(("").join(r))
            if not i==len(st):
                r.append(st[i:])
        return r
        
    @timethis
    def get_segwords(self, df_ngram):
        df_segwords = pd.DataFrame()
        suffixs = ['_unigram', '_bigram', '_trigram']
        for suffix in suffixs:
            df_segwords["q|t"+suffix] = df_ngram.apply(lambda x: \
                    self._seg_words(x["q"+suffix], x["t"+suffix]), axis=1)
        return df_segwords
          
    def _str_whole_word(self, str1, str2):
        if str1 == '' or str2 == '':
            return 0  
        cnt, pos = 0, 0
        while True:
            pos = str2.find(str1, pos)
            if pos == -1:
                return cnt
            else:
                pos += len(str1)
                cnt += 1
    
    def _str_common_word(self, str1, str2):
        if str1 == '' or str2 == '':
            return 0        
        words, cnt = str1.split(), 0
        for word in words:
            if str2.find(word)>=0:
                cnt+=1
        return cnt
    
    def _str_last_word_common(self, str1, str2):
        if str1 == '' or str2 == '':
            return 0    
        word = str1.split()[-1]
        cnt, pos = 0, 0
        while True:
            pos = str2.find(word, pos)
            if pos == -1:
                return cnt
            else:
                pos += len(word)
                cnt += 1
    
    @timethis
    def extract_rfrin_feats(self, df_join_ng):
        """
        Extract in features.
        
        Returns:
            df_feat(pd.DataFrame)
        """        
        df_feat = pd.DataFrame()
        suffixs = ['_unigram', '_bigram', '_trigram'] # better use all
        for suffix in suffixs:
            # Count of words of query in title, description
            df_feat["q|t"+suffix] = df_join_ng.apply(lambda x: \
                    self._str_whole_word(x["q"+suffix+"_join"], x["t"+suffix+"_join"]), axis=1)
            df_feat["q|d"+suffix] = df_join_ng.apply(lambda x: \
                    self._str_whole_word(x["q"+suffix+"_join"], x["d"+suffix+"_join"]), axis=1)
            # Count of last words of query in title, description
            df_feat["q_last|t"+suffix] = df_join_ng.apply(lambda x: \
                self._str_last_word_common(x["q"+suffix+"_join"], x["t"+suffix+"_join"]), axis=1)
            df_feat["q_last|d"+suffix] = df_join_ng.apply(lambda x: \
                self._str_last_word_common(x["q"+suffix+"_join"], x["d"+suffix+"_join"]), axis=1)            
        df_rfrin_feats = df_feat 
        saveit(df_rfrin_feats, 'df_rfrin_feats')
        
    @timethis
    def extract_brand_feats(self, df_ngram, df_data, df_wordcountfeats):
        """
        Extract brand features.
        
        Returns:
            df_feat(pd.DataFrame)        
        """
        # Generate segwords
        df_segwords = self.get_segwords(df_ngram)  
        # segwords and brand
        df_merge = pd.merge(df_segwords, df_data, left_index=True, right_index=True)
        suffixs = ['_unigram'] # use '_bigram' and '_trigram' do not change result
        df_feat = pd.DataFrame()
        for suffix in suffixs:
            df_feat['word|b'+suffix] = df_merge.apply(lambda x: self._str_common_word\
                                                    (x["q|t"+suffix], x['b']), axis=1)
            df_feat['ratio_b'+suffix] = df_feat['word|b'+suffix].div\
                                            (df_wordcountfeats['b'+suffix+'_listlen']).fillna(0)
        # brand factor
        df_brand = pd.unique(df_data.b.ravel())
        d={}
        i = 1000
        for s in df_brand:
            d[s]=i
            i+=3
        df_feat["b_factor"] = df_data['b'].map(lambda x:d[x])
        df_brand_feats = df_feat
        saveit(df_brand_feats, 'df_brand_feats')
        
    @timethis
    def extract_tfidf_feats(self, df_data, n_components):
        """
        Extract tfidf features.
        
        Returns:
            df_feat(pd.DataFrame)        
        """        
        df_feat = pd.DataFrame(index=range(df_data.shape[0]))
        tfidf = TfidfVectorizer(ngram_range=(2, 4), stop_words='english')
        tsvd = TruncatedSVD(n_components=n_components, random_state = 2016)
        df_data['q'].to_csv('q')
        df_data['t'].to_csv('t')
        df_data['d'].to_csv('d')
        tfidf.set_params(input='filename')        
        tfidf.fit(['q','t','d'])
        tfidf.set_params(input='content')  
        for col in ['q', 't', 'd', 'b']:
            print('process column', col)
            txt = df_data[col]
            tfidf_mat = tfidf.transform(txt)
            nd_feat = tsvd.fit_transform(tfidf_mat)
            tmp = pd.DataFrame(nd_feat, columns=[col+'_tfidf_comp'+str(i) \
                                        for i in range(n_components)])
            df_feat = pd.merge(df_feat, tmp, left_index=True, right_index=True)
        df_tfidf_feats = df_feat
        saveit(df_tfidf_feats, 'df_tfidf_feats')
        
        
class SECFeatureGenerator(object):
    """
    Generate SEC features.
    Reference the public code of the 2nd place winner of kaggle 'crowdflower' competition.
    """    
    def __init__(self, config):
        self.w2v_path = config.W2V_PATH
    
    def _compression_distance(self, x, y, l_x=None, l_y=None):
        x, y = str(x), str(y)
        if x==y:
            return 0
        x_b = x.encode('utf-8')
        y_b = y.encode('utf-8')
        if l_x is None:
            l_x = len(lzma.compress(x_b))
            l_y = len(lzma.compress(y_b))
        l_xy = len(lzma.compress(x_b+y_b))
        l_yx = len(lzma.compress(y_b+x_b))
        dist = (min(l_xy,l_yx)-min(l_x,l_y))/max(l_x,l_y)
        return dist
    
    @timethis
    def extract_compressdist_feat(self, df_data):
        """
        Extract compression distance features.      
        """
        df_feat = pd.DataFrame(index=range(df_data.shape[0]))
        compdist_func = lambda x: self._compression_distance(x['q'], x['t'])
        df_feat['dist_qt'] = df_data.apply(compdist_func, axis=1)
        seqdist_func = lambda x: 1 - SequenceMatcher(None, x['q'], x['t']).ratio()
        df_feat['dist_qt2'] = df_data.apply(seqdist_func, axis=1)
        df_compressdist_feats = df_feat
        saveit(df_compressdist_feats, 'df_compressdist_feats')
    
    @timethis
    def extract_seq_related_feat(self, df_data):
        """
        Extract SequenceMatcher related features besides compression distance.    
        """
        list_word_counter_qt = []
        list_lev_max = []
        list_last_word_in = []
        list_word_counter_qt_norm =[]
        
        # process each row
        for ind in df_data.index: 
            query, title = df_data.loc[ind,'q'], df_data.loc[ind,'t']
            query_len = len(query.split())
            word_counter_qt = 0
            lev_dist_arr = []
            for q in query.split():
                lev_dist_q = []
                for t in title.split():
                    lev_dist = SequenceMatcher(None,q,t).ratio()
                    if lev_dist > 0.9:
                        word_counter_qt += 1
                        #tmp_title += ' '+q # add such words to title to increase their weights in tfidf
                    lev_dist_q.append(lev_dist)
                lev_dist_arr.append(lev_dist_q)
            last_word_in = 0
            for t in title.split():
                lev_dist = SequenceMatcher(None,query.split()[-1],t).ratio()
                if lev_dist > 0.9: 
                    last_word_in = 1
            lev_max = 0
            for item in lev_dist_arr:
                lev_max_q = max(item)
                lev_max += lev_max_q
            lev_max = 1- lev_max/len(lev_dist_arr)
            word_counter_qt_norm = word_counter_qt/query_len
            
            # Record
            list_word_counter_qt.append(word_counter_qt)
            list_lev_max.append(lev_max)
            list_last_word_in.append(last_word_in)
            list_word_counter_qt_norm.append(word_counter_qt_norm)
        
        df_feat = pd.DataFrame(index=df_data.index.values)
        df_feat['word_counter_qt'] = pd.Series(list_word_counter_qt, index=df_data.index.values)
        df_feat['lev_max'] = pd.Series(list_lev_max, index=df_data.index.values)
        df_feat['last_word_in'] = pd.Series(list_last_word_in, index=df_data.index.values)
        df_feat['word_counter_qt_norm'] = pd.Series(list_word_counter_qt_norm, \
                                                    index=df_data.index.values)
        df_seq_feats = df_feat
        saveit(df_seq_feats, 'df_seq_feats')
        
    def _cosine_dist(self, text_a ,text_b, vect):
        ta, tb = vect.transform([text_a]), vect.transform([text_b])
        return pairwise_distances(ta, tb, metric='cosine')[0][0]    
        
    @timethis
    def extract_cosinedist_feat(self, df_data, df_w2vlem_join):
        """
        Extract cosine distance features.
        Note: this func is very slow. It would cost a few hours.       
        """
        df_feat = pd.DataFrame(index=df_data.index.values)
        tfv = TfidfVectorizer(ngram_range=(2,3), min_df=2)
        
        print('computing qt_w2v_cosdist')
        df_w2vlem_join['q_w2v'].to_csv('q_w2v', index=False)
        df_w2vlem_join['t_w2v'].to_csv('t_w2v', index=False)
        tfv.set_params(input='filename')
        tfv.fit(['q_w2v', 't_w2v'])# list(df_w2vlem_join['q_w2v'].values)+list(df_w2vlem_join['t_w2v'].values)
        tfv.set_params(input='content')        
        print('done fitting')
        qt_unigram_func = lambda x: self._cosine_dist(x['q_w2v'], x['t_w2v'], tfv)
        df_feat['qt_w2v_cosdist'] = df_w2vlem_join.apply(qt_unigram_func, axis=1)   
        
        if True: # You can abandon this feature because it costs too much time (more than 24 hours). 
            print('computing qd_w2v_cosdist')
            df_w2vlem_join['d_w2v'].to_csv('d_w2v', index=False)
            tfv.set_params(input='filename')
            tfv.fit(['q_w2v', 'd_w2v'])# list(df_w2vlem_join['q_w2v'].values)+list(df_w2vlem_join['d_w2v'].values)
            tfv.set_params(input='content')
            print('done fitting')
            qd_unigram_func = lambda x: self._cosine_dist(x['q_w2v'], x['d_w2v'], tfv)
            df_feat['qd_w2v_cosdist'] = df_w2vlem_join.apply(qd_unigram_func, axis=1)   
        
        print('computing qt_cosdist')
        df_data['q'].to_csv('q', index=False)
        df_data['t'].to_csv('t', index=False)
        tfv.set_params(input='filename')
        tfv.fit(['q', 't'])# list(df_data['q'].values) + list(df_data['t'].values)
        tfv.set_params(input='content')
        print('done fitting')
        qt_func = lambda x: self._cosine_dist(x['q'], x['t'], tfv)
        df_feat['qt_cosdist'] = df_data.apply(qt_func, axis=1)              
        
        df_cosdist_feats = df_feat 
        saveit(df_cosdist_feats, 'df_cosdist_feats')   
        
    def _calc_w2v_sim(self, row, embedder):
        """
        Calc w2v similarities and diff of centers of query\title
        """
        for x in row['q_w2v']:
            print(x)
        a2 = [x for x in row['q_w2v'] if x in embedder.vocab]
        b2 = [x for x in row['t_w2v'] if x in embedder.vocab]
        if len(a2)>0 and len(b2)>0:
            w2v_sim = embedder.n_similarity(a2, b2)
        else:
            return -1, -1, np.zeros(300)
        
        vectorA = np.zeros(300)
        for w in a2:
            vectorA += embedder[w]
        vectorA /= len(a2)

        vectorB = np.zeros(300)
        for w in b2:
            vectorB += embedder[w]
        vectorB /= len(b2)

        vector_diff = (vectorA - vectorB)

        w2v_vdiff_dist = np.sqrt(np.sum(vector_diff**2))
        return w2v_sim, w2v_vdiff_dist, vector_diff
        
    @timethis
    def extract_w2v_feat(self, df_w2vlem):
        """
        Extract word2vec similarities and diff of centers features between query and title.      
        """    
        embedder = Word2Vec.load_word2vec_format(self.w2v_path, binary=True)
        
        sim_list = []
        dist_list = []
        for i, row in df_w2vlem.iterrows():
            sim, dist, _ = self._calc_w2v_sim(row, embedder)
            sim_list.append(sim)
            dist_list.append(dist)

        df_feat = pd.DataFrame(index=df_w2vlem.index.values)
        df_feat['w2v_sim'] = np.array(sim_list)
        df_feat['w2v_dist'] = np.array(dist_list)
        df_w2v_feats = df_feat
        saveit(df_w2v_feats, 'df_w2v_feats')
        
    @timethis
    def extract_tsne_feat(self, df_w2vlem_join):
        """
        Extract tsne features.
        Note: MemoryError      
        """  
        ts = TSNE(2)
        
        df_feat = pd.DataFrame(index=df_w2vlem_join.index.values)
        
        vect = TfidfVectorizer(ngram_range=(1,2), min_df=3)
        X_t = vect.fit_transform(df_w2vlem_join['t_w2v'].tolist())
        svd = TruncatedSVD(n_components=100, random_state=2016)
        X_svd = svd.fit_transform(X_t)
        print(X_svd.shape)
        X_scaled = StandardScaler().fit_transform(X_svd)
        print(X_scaled.shape)
        X_tsne = ts.fit_transform(X_scaled)
        df_feat['tsne_t_1'] = X_tsne[:len(df_w2vlem_join), 0]
        df_feat['tsne_t_2'] = X_tsne[:len(df_w2vlem_join), 1]
        
        vect = TfidfVectorizer(ngram_range=(1,2), min_df=3)
        X_q = vect.fit_transform(df_w2vlem_join['q_w2v'].tolist())
        X_tq = sp.hstack([X_t, X_q]).tocsr()
        svd = TruncatedSVD(n_components=100)
        X_svd = svd.fit_transform(X_tq)
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = ts.fit_transform(X_scaled)
        df_feat['tsne_qt_1'] = X_tsne[:len(df_w2vlem_join), 0]
        df_feat['tsne_qt_2'] = X_tsne[:len(df_w2vlem_join), 1]
        
        vect = TfidfVectorizer(ngram_range=(1,2), min_df=3)
        X_d = vect.fit_transform(df_w2vlem_join['d_w2v'].tolist())
        svd = TruncatedSVD(n_components=100)
        X_svd = svd.fit_transform(X_d)
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = ts.fit_transform(X_scaled)
        df_feat['tsne_desc_1'] = X_tsne[:len(df_w2vlem_join), 0]
        df_feat['tsne_desc_2'] = X_tsne[:len(df_w2vlem_join), 1]
        
        df_tsne_feats = df_feat
        saveit(df_tsne_feats, 'df_tsne_feats')
        
