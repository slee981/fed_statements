#######################################################################
# Goal   : take dataframe with date | speaker | text and add a column
#          with the 1, 2, and 3 gram versions

# Author : Stephen Lee 
# Date   : 9.23.19
#######################################################################

import pandas as pd 
from gensim.parsing.porter import PorterStemmer
from datetime import datetime
import os, sys 

###########################################################################
# Storage
###########################################################################

ROOT_DIR = os.getcwd()
DATA_DIR = 'data'
DF_FILE = os.path.join(ROOT_DIR, DATA_DIR, 'speeches.csv')
STOP_WORDS = os.path.join(ROOT_DIR, 'code', 'clean', 'stopwords.txt')
FINAL_DF_FILE = os.path.join(ROOT_DIR, DATA_DIR, 'speeches_ngrams.csv')

df = pd.read_csv(DF_FILE)
ps = PorterStemmer()

###########################################################################
# Functions
###########################################################################

def only_alphas(txt):
    txt = str(txt)
    txt = txt.replace('Watch Live', '').replace('\n', ' ')
    t = [i.lower() for i in txt if i.isalpha() or i == ' ']
    return ''.join(t)

def get_stop_words():
    with open(STOP_WORDS, 'r') as f: 
        txt = f.read()
    return txt.split('\n')

def remove_stop_words(txt, stop_words): 
    txt = txt.split(' ')
    return ' '.join([w.strip() for w in txt if ((w not in stop_words) and (w != ' ') and (w != '') and len(w) > 2) ])

def stem(txt): 
    global ps
    txt = txt.split(' ')
    return ' '.join([ps.stem(w) for w in txt])

def ngrams(txt, n_gram=1):
    '''
    INPUT   : txt    -> text in alpha only, stemmed, form
            : n_gram -> the length of each phrase to generate
    OUTPUT  : a string of '.' separated ngram phrases
    '''
    token = [t for t in txt.split(' ')]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    ngrams_lst = [' '.join(g) for g in ngrams]
    return '.'.join(ngrams_lst)

###########################################################################
# Main
###########################################################################

if __name__ =='__main__': 

     # read raw text data 
    print('Reading data...')
    df = pd.read_csv(DF_FILE).drop('Unnamed: 0', axis=1)

    # remove all non alpha characters 
    print('Removing all non alpha characters...')
    df['only_alpha'] = df['Text'].apply(only_alphas)

    # remove stopwords 
    print('Removing stop words...')
    stop_words = get_stop_words()
    df['no_stop'] = df['only_alpha'].apply(remove_stop_words, args=[stop_words])
    df = df.drop('only_alpha', axis=1)

    # stem 
    print('Stemming...')
    df['stemmed'] = df['no_stop'].apply(stem)
    df = df.drop('no_stop', axis=1)
    
    # generate n-grams
    print('Generating ngrams...')
    df['1-gram'] = df['stemmed'].apply(ngrams, args=[1])
    df['2-gram'] = df['stemmed'].apply(ngrams, args=[2])
    df['3-gram'] = df['stemmed'].apply(ngrams, args=[3])

    # save 
    print('Saving...')
    df.to_csv(FINAL_DF_FILE)
    print('Saved.')