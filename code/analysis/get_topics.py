###########################################################################
# Notes
###########################################################################
# Get topics from one of the saved LDA models


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
NUM_TOPICS = 15

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DF_WITH_LDA = os.path.join(DATA_DIR, '{}_topic_lda.csv'.format(NUM_TOPICS))
LDA_FILE_NAME = os.path.join(DATA_DIR, 'models', '{}_topic_lda'.format(NUM_TOPICS))

###########################################################################
# Functions
###########################################################################

def print_topics(lda_model): 
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic), end='\n\n')

###########################################################################
# Main
###########################################################################

# load model 
lda = gensim.models.LdaMulticore.load(LDA_FILE_NAME)
print_topics(lda)

# load data 
df = pd.read_csv(DF_WITH_LDA).drop('Unnamed: 0', axis=1)