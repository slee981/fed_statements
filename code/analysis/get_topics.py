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
import os, sys

###########################################################################
# Storage
###########################################################################

# specs
NUM_TOPICS = 10

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data")
DF_WITH_LDA = os.path.join(
    DATA_DIR, "lda_dataframes", "{}_topic_lda.csv".format(NUM_TOPICS)
)
LDA_FILE_NAME_TEMPLATE = os.path.join(DATA_DIR, "lda_models", "{}_topic_lda")

###########################################################################
# Functions
###########################################################################


def print_topics(lda_model):
    for idx, topic in lda_model.print_topics(-1):
        print("\nTopic: {} \nWords: {}".format(idx, topic), end="\n\n")


###########################################################################
# Main
###########################################################################

if len(sys.argv) == 2 and isinstance(sys.argv[1], int):
    fname = LDA_FILE_NAME_TEMPLATE.format(sys.argv[1])
else:
    fname = LDA_FILE_NAME_TEMPLATE.format(NUM_TOPICS)

# load model
lda = gensim.models.LdaMulticore.load(fname)
print_topics(lda)

# load data
df = pd.read_csv(DF_WITH_LDA).drop("Unnamed: 0", axis=1)

