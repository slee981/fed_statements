###########################################################################
# Notes
###########################################################################
# Get topics from one of the saved LDA models and save to csv


###########################################################################
# Imports
###########################################################################

import pandas as pd
import numpy as np
import gensim
from collections import defaultdict
import os, sys, csv

###########################################################################
# Storage
###########################################################################

# specs
NUM_TOPICS = 12

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data")
DF_WITH_LDA = os.path.join(
    DATA_DIR, "lda_dataframes", "{}_topic_lda.csv".format(NUM_TOPICS)
)
LDA_FILE_NAME_TEMPLATE = os.path.join(DATA_DIR, "lda_models", "{}_topic_lda")
CSV_FILE_NAME = os.path.join(
    ROOT_DIR, "figures", "words_{}_topics.csv".format(NUM_TOPICS)
)

###########################################################################
# Functions
###########################################################################


def write(lst_of_lsts, fname):
    with open(fname, "a") as f:
        writer = csv.writer(f)
        for lst in lst_of_lsts:
            writer.writerow(lst)


def topics_to_csv(lda, fname):
    topics = []
    for idx, topic in lda.print_topics(-1):
        topicnum = "Topic {}".format(idx)
        row = [topicnum]
        for wt_wd in topic.split("+"):
            wt, word = wt_wd.split("*")
            word = word.replace('"', "")
            wt = float(wt)
            row.append(word)
        topics.append(row)
    write(topics, fname)


###########################################################################
# Main
###########################################################################

lda_fname = LDA_FILE_NAME_TEMPLATE.format(NUM_TOPICS)
csv_fname = CSV_FILE_NAME

# load model
lda = gensim.models.LdaMulticore.load(lda_fname)
topics_to_csv(lda, csv_fname)

