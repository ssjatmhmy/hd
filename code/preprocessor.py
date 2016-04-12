from util import timethis
import pandas as pd 
import numpy as np
from collections import Counter
import ngram
import os
import csv
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import wordnet


class PreProcessor(object):
    """
    Clean up the data; do some data format transformation.
    """
    def __init__(self, config):
        self._clean_func = config.clean_func
        # Load wordmap
        wordmap_path = os.path.join(config.DATA_PATH, 'wordmap.csv')
        self.wordmap = self._load_wordmap(wordmap_path)
        # Load stopwords
        self.stopwords = nltk.corpus.stopwords.words("english")
        # Load stemmer
        if config.STEM_TYPE == 'porter':
            self.stemmer = nltk.stem.PorterStemmer()
        elif config.STEM_TYPE == "snowball":
            self.stemmer = nltk.stem.SnowballStemmer('english')
        else:
            assert True, "Configuration Error: Unknown stemmer."

    def _load_wordmap(self, wordmap_path):
        """
        Return a wordmap that maps misspelled words, synonyms, etc to correct/consolidated words.
        """
        wordmap = {}
        with open(wordmap_path) as f:
            for line in csv.reader(f):
                if line[0].startswith('#'):
                    continue        
                word, mapped_word = line
                wordmap[word] = mapped_word
        return wordmap
        
    def _replace(self, words):
        for word in words.split():
            yield self.wordmap.get(word, word) 
        
    def _clean_line(self, line):
        l = line.lower()
        # replace words
        l = self._replace(l)
        line = " ".join(l)
        return line
        
    @timethis
    def clean_text(self, df_data):
        """
        Replace misspelled words, synonyms, etc with correct/consolidated words. Turn all letters \
        into lower case. 
        Note that this is an inplace operation (i.e., modified df_data).
        """
        df_data = df_data.fillna('')  
        for column in ["q", "t", "d", "b", "value"]:
            try:
                # The first cleaning.
                # Call user-defined clean function: clean_func. This function is defined in config.py.            
                df_data[column] = df_data[column].map(self._clean_func)     
                # The second cleaning.
                # Turn all letters into lower case and replace some words with with \
                # correct/consolidated words. The word map used in this replacement is read from file \
                # 'wordmap.csv' in data folder.
                df_data[column] = df_data[column].map(self._clean_line)           
            except KeyError:
                continue   
        return df_data.fillna('')

    def _stem_excl_words(self, words):
        """
        Stem words and exclude stopwords
        """
        for w in words.split():
            if w not in self.stopwords:
                yield self.stemmer.stem(w) 
            
    def _get_ngram(self, sr):
        """
        Compute ngram of the text of a pd.Series. The unigram operation is combining stemming \
        words and excluding stopwords. The bigram and trigram operations are based on the results \
        of the unigram operation.
        
        Args:
            sr(pd.Series):
            
        Returns:
            sr_unigram(pd.Series), sr_bigram(pd.Series), sr_trigram(pd.Series)
        """
        # Unigram.
        unigram_func = lambda s: list(self._stem_excl_words(s))
        sr_unigram = sr.map(unigram_func)     
        # Bigram.
        bigram_func = lambda s: ngram.getBigram(s, '_')
        sr_bigram = sr_unigram.map(bigram_func) 
        # Trigram.
        trigram_func = lambda s: ngram.getTrigram(s, '_')
        sr_trigram = sr_unigram.map(trigram_func) 
        return sr_unigram, sr_bigram, sr_trigram

    @timethis
    def get_ngram(self, df):
        """
        Return unigram, bigram, trigram columns corresponding to the columns of df that can be used\
        to generate ngram. 
        
        Args:
            df(pd.DataFrame)
            
        Returns:
            df_ngram(pd.DataFrame)
        """
        df_ngram = pd.DataFrame()
        # Only process the columns that can be used to generate ngram.
        for column in ["q", "t", "d", "b"]:
            try:
                sr = df[column]
            except KeyError:
                continue
            # Add ngram.
            cols = (column+'_unigram', column+'_bigram', column+'_trigram')
            df_ngram[cols[0]], df_ngram[cols[1]], df_ngram[cols[2]] = self._get_ngram(sr)
        return df_ngram    
        
    @timethis
    def join_ngram(self, df_ngram):
        df_join_ng = pd.DataFrame()
        for col in df_ngram.columns:
            df_join_ng[col+'_join'] = df_ngram[col].map(lambda x: ' '.join(x))
        return df_join_ng
        
    def _scale(self, nd_array):
        nd_scale = nd_array.copy()
        for val, i in enumerate(nd_scale):
            if nd_scale[i] <= 1.5:
                nd_scale[i] = 1
            elif nd_scale[i] <= 2.0:
                nd_scale[i] = 2
            elif nd_scale[i] <= 2.5:
                nd_scale[i] = 3
            else:
                nd_scale[i] = 4
        return nd_scale
                   
    @timethis
    def construct_extended_query(self, df_data, n_train, nd_label, top_words=10):
        """
        Construct extended query. 
            
        Returns:
            df_extq(pd.DataFrame)
        """
        y = self._scale(nd_label)
        
        train, test = df_data[:n_train], df_data[n_train:]
        
        queries, queries_test = train['q'].values, test['q'].values
        titles = train['t'].values
        
        query_ext_train = train['q'].values #np.zeros(len(train)).astype(np.object)
        query_ext_test = test['q'].values #np.zeros(len(test)).astype(np.object)
        for q in np.unique(queries):
            q_mask = queries == q
            q_test = queries_test == q
            
            titles_q = titles[q_mask]
            y_q = y[q_mask]
            
            good_mask = y_q > 3
            titles_good = titles_q[good_mask]
            ext_q = str(q)
            for item in titles_good:
                ext_q += ' '+str(item)
            c = [word for word, it in Counter(ext_q.split()).most_common(top_words)]
            c = ' '.join(c)
            query_ext_train[q_mask] = c
            query_ext_test[q_test] = c
        
        df_extq = pd.DataFrame(index=df_data.index.values)
        df_extq['extq'] = np.hstack((query_ext_train,query_ext_test))
        
        return df_extq.fillna('')
        
    def _w2vlem_words(self, s, toker, lemmer):
        tokens = toker.tokenize(s)
        return [lemmer.lemmatize(z) for z in tokens]
    
    def _join_w2vlem_words(self, s, toker, lemmer):
        return " ".join(self._w2vlem_words(s, toker, lemmer))
        
    @timethis
    def w2vlem(self, df_data):
        """
        Construct Word2Vec lemmatized query and title. 
            
        Returns:
            df_w2vqt(pd.DataFrame)
        """ 
        toker = TreebankWordTokenizer()
        lemmer = wordnet.WordNetLemmatizer()        
        w2v_func = lambda s: self._w2vlem_words(s, toker, lemmer)
        df_w2vqt = pd.DataFrame(index=df_data.index.values)
        df_w2vqt['q_w2v'] = df_data['q'].map(w2v_func)
        df_w2vqt['t_w2v'] = df_data['t'].map(w2v_func)
        df_w2vqt['d_w2v'] = df_data['d'].map(w2v_func)
        return df_w2vqt
        
    @timethis
    def join_w2vlem(self, df_data):
        """
        Construct join version of Word2Vec lemmatized query and title. 
            
        Returns:
            df_w2vqt_join(pd.DataFrame)
        """ 
        toker = TreebankWordTokenizer()
        lemmer = wordnet.WordNetLemmatizer()        
        w2v_join_func = lambda s: self._join_w2vlem_words(s, toker, lemmer)
        df_w2vqt_join = pd.DataFrame(index=df_data.index.values)
        df_w2vqt_join['q_w2v'] = df_data['q'].map(w2v_join_func)
        df_w2vqt_join['t_w2v'] = df_data['t'].map(w2v_join_func)
        df_w2vqt_join['d_w2v'] = df_data['d'].map(w2v_join_func)
        return df_w2vqt_join
        
