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
MIN_SPEECH_LENGTH = 50
N = 2
N_GRAMS = '{}-gram'.format(N)
VOCAB_SIZE = 15000
NUM_TOPICS = 40
LOOPS = 5
PCT_TOPIC_THRESHOLD = 0.05

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CLEAN_DATA_FILE = os.path.join(DATA_DIR, 'master_df.csv')
DF_WITH_LDA = os.path.join(DATA_DIR, 'dates_{}_topic_lda_{}_loop.csv'.format(NUM_TOPICS, LOOPS))
LDA_FILE_NAME_TEMPLATE = os.path.join(DATA_DIR, 'models', '{}_topic_lda_{}_loop')

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

def get_num_topics_per_period(loops = 1): 
    global df, corpus, vocab_dict, lda_model, NUM_TOPICS

    topics_per_section = [[] for i in range(len(corpus))]
    dates = df['Date'].copy()
    
    for loop in range(loops):

        print('Fitting model...')
        lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=vocab_dict, num_topics=NUM_TOPICS, passes=2, workers=2)

        print('Saving model...')
        lda_fname = LDA_FILE_NAME_TEMPLATE.format(NUM_TOPICS, loop + 1)
        lda_model.save(lda_fname)
        
        # to reload...
        # lda = gensim.models.LdaMulticore.load(LDA_FILE_NAME)
        
        # find best topic for each speech period
        for i, row in enumerate(lda_model[corpus]):

            # count topics with percent contribution > threshold
            n_topics = 0
            for topic_num, pct_topic in row: 
                
                if pct_topic > PCT_TOPIC_THRESHOLD: 
                    n_topics += 1 

            topics_per_section[i].append(n_topics)

    # find the average number of topics
    topics_per_section = list(map(np.mean, topics_per_section))

    data = {
        'Date': dates, 
        'Num Topics': topics_per_section
    }

    return pd.DataFrame(data)

###########################################################################
# Main
###########################################################################

if __name__ == '__main__':

    # read data
    print('Reading data...')
    df = pd.read_csv(CLEAN_DATA_FILE).drop('Unnamed: 0', axis=1)
    dates = df['Date'].copy()
   
    # get ngrams, build vocab, and bag of ngrams
    doc_ngrams = get_ngrams(df)
    vocab_dict = gensim.corpora.Dictionary(doc_ngrams)
    vocab_dict.filter_extremes(no_below=15, no_above=0.25, keep_n=VOCAB_SIZE)
    corpus = [vocab_dict.doc2bow(doc) for doc in doc_ngrams]

    topics_per_period = get_num_topics_per_period(loops=LOOPS)
    df = df.merge(topics_per_period, on='Date')
    print(df[['Date', 'Num Topics']].head(10))

    print('Saving dataframe...')
    df.to_csv(DF_WITH_LDA)
    print('Saved.')
