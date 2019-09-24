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
import numpy as np
import gensim
from collections import defaultdict
import os

###########################################################################
# Storage
###########################################################################

# specs 
NUM_TOPICS = 40
LOOPS = 1

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DF_WITH_LDA = os.path.join(DATA_DIR, 'dates_{}_topic_lda_{}_loop.csv'.format(NUM_TOPICS, LOOPS))
LDA_FILE_NAME = os.path.join(DATA_DIR, 'models', '{}_topic_lda'.format(NUM_TOPICS))

###########################################################################
# Functions
###########################################################################

def print_topics(lda_model): 
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic), end='\n\n')

# load model 
lda = gensim.models.LdaMulticore.load(LDA_FILE_NAME)
print_topics(lda)