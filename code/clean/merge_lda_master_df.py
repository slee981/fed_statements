###########################################################################
# Notes
###########################################################################
# LDA analysis tries to find latent topics and the words that are
# associated with them.

###########################################################################
# Imports
###########################################################################

import pandas as pd 
import numpy as np
import gensim
from collections import defaultdict
import os

###########################################################################
# Storage
###########################################################################

# specs 
NUM_TOPICS = [i for i in range(9, 21)]
PCT_TOPIC_THRESHOLD = 0.05 
ETA = 0.025

N = 2
N_GRAMS = '{}-gram'.format(N)
VOCAB_SIZE = 15000
MIN_SPEECH_LENGTH = 50

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MASTER_DF = os.path.join(DATA_DIR, 'master_df.csv')
DF_WITH_LDA_TEMPLATE = os.path.join(DATA_DIR, '{}_topic_lda.csv')
LDA_FILE_NAME_TEMPLATE = os.path.join(DATA_DIR, 'models', '{}_topic_lda')

###########################################################################
# Functions
###########################################################################

def get_ngrams(df): 
    def ngram_str_to_lst(ngram_str):
        if isinstance(ngram_str, float): 
            ngram_str = ''
        return ngram_str.split('.')
    col = df[N_GRAMS].copy()
    return col.apply(ngram_str_to_lst)

def print_topics(): 
    global lda_model
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic), end='\n\n')

def get_num_topics_per_period(num_topics, alpha, eta): 
    global df, corpus, vocab_dict, lda_model

    topics_dist_per_section = [[0 for i in range(num_topics)] for i in range(len(corpus))]
    topics_per_section = []
    dates = df['Date'].copy()

    print('Fitting model with {} topics...'.format(num_topics))
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=vocab_dict,
                                           num_topics=num_topics, passes=2, workers=2, 
                                           alpha=alpha, eta=eta)

    print('Saving model...')
    lda_fname = LDA_FILE_NAME_TEMPLATE.format(num_topics)
    lda_model.save(lda_fname)
    
    # to reload...
    # lda = gensim.models.LdaMulticore.load(LDA_FILE_NAME)
    
    # find best topic for each speech period
    for i, row in enumerate(lda_model[corpus]):

        # count topics with percent contribution > threshold
        n_topics = 0
        for topic_num, pct_topic in row: 
            topics_dist_per_section[i][topic_num] = pct_topic
            if pct_topic > PCT_TOPIC_THRESHOLD: 
                n_topics += 1 

        topics_per_section.append(n_topics)

    data = {
        'Date': dates, 
        'Num Topics': topics_per_section, 
        'Topic Dist': topics_dist_per_section
    }

    return pd.DataFrame(data)

###########################################################################
# Main
###########################################################################

if __name__ == '__main__':

    # read data
    print('Reading data...')
    df = pd.read_csv(MASTER_DF).drop('Unnamed: 0', axis=1)
    dates = df['Date'].copy()

    # get ngrams, build vocab, and bag of ngrams
    doc_ngrams = get_ngrams(df)
    vocab_dict = gensim.corpora.Dictionary(doc_ngrams)
    vocab_dict.filter_extremes(no_below=15, no_above=0.25, keep_n=VOCAB_SIZE)
    corpus = [vocab_dict.doc2bow(doc) for doc in doc_ngrams]

    for n in NUM_TOPICS: 
        alpha = 50 / n      # <- recommended values per Griffiths and Steyvers 2004 
        eta = 0.025
        fname = DF_WITH_LDA_TEMPLATE.format(n)
        topics_per_period = get_num_topics_per_period(num_topics = n, alpha = alpha, eta = eta)
        df_final = df.merge(topics_per_period, on='Date')
        print(df_final.columns)
        print(df_final[['Date', 'Topic Dist']].head(5))

        print('Saving dataframe...')
        df_final.to_csv(fname)
        print('Saved.')
