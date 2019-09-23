###########################################################################
# Notes
###########################################################################
# LDA analysis tries to find latent topics and the words that are
# associated with them. Many ways to spin it, but so far one interesting 
# bit is that with the following params: 
#
# VOCAB_SIZE = 15000
# LDA_SUBSET = True
# SOURCE_ONE = 'Vox'
# SOURCE_TWO = 'Fox'
#
# We get two arguably different topics: 
# - taxes and healthcare 
# - boarder wall, border security, etc 


###########################################################################
# Imports
###########################################################################

import pandas as pd 
# import pyLDAvis
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

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CLEAN_DATA_FILE = os.path.join(DATA_DIR, 'speeches_ngrams.csv')

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

###########################################################################
# Main
###########################################################################

if __name__ == '__main__':

    # read data
    print('Reading data...')
    df = pd.read_csv(CLEAN_DATA_FILE).drop('Unnamed: 0', axis=1).set_index('Date')
   
    # get ngrams, build vocab, and bag of ngrams
    doc_ngrams = get_ngrams(df)
    vocab_dict = gensim.corpora.Dictionary(doc_ngrams)
    vocab_dict.filter_extremes(no_below=15, no_above=0.25, keep_n=VOCAB_SIZE)
    corpus = [vocab_dict.doc2bow(doc) for doc in doc_ngrams]

    # look at contents and mappings
    # print(df['articles_stemmed'].iloc[1233])
    # doci = bag_of_ngrams[1233]
    # for j in range(len(doci)):
    #     print("Word {} (\"{}\") appears {} time.".format(doci[j][0], vocab_dict[doci[j][0]], doci[j][1]))

    print('Fitting model...')
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=vocab_dict, num_topics=NUM_TOPICS, passes=2, workers=2)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic), end='\n\n')

    # find best topic for each speech 
    for i, row in enumerate(lda_model[corpus[10:12]]):
        print(i)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row): 
            print(topic_num, prop_topic)
            wp = lda_model.show_topic(topic_num)
            print(', '.join([word for word, prop in wp]))      

    #print(df['Text'].iloc[10])