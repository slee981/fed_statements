import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import gensim
import numpy as np
from random import sample
from math import floor
import os
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical
from sklearn import metrics

###########################################################################
# Storage
###########################################################################

# specs
NUM_TOPICS = range(9, 16)
N_CATEGORIES = 3

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data")
DF_WITH_LDA_TEMPLATE = os.path.join(DATA_DIR, "lda_dataframes", "{}_topic_lda.csv")
LDA_FILE_NAME_TEMPLATE = os.path.join(DATA_DIR, "lda_models", "{}_topic_lda")

###########################################################################
# Functions
###########################################################################


def get_topic_dataframe(df):
    def str_to_arr(str_arr):
        return [
            float(ele) for ele in str_arr.replace("[", "").replace("]", "").split(",")
        ]
    topic_dists = df.copy()
    rows = df.shape[0]
    num_topics = len(str_to_arr(topic_dists.iloc[0]))
    x = []
    cols = ["Topic{}".format(i) for i in range(num_topics)]
    for r in range(rows):
        obs = str_to_arr(topic_dists.iloc[r])
        x.append(obs)
    return pd.DataFrame(x, columns=cols)


def classify_changes(change):
    if change == 0:  # no change
        return 0
    elif change > 0:  # increase rates
        return 1
    elif change < 0:  # decrease rates
        return 2
    else:
        raise TypeError


def split_train_test(df, train_pct=0.8):
    """
    INPUT: 
        df                     ~ pandas dataframe, all data 
        train_pct              ~ int, pct to split into training. 
    OUTPUT:           
        (train_df, test_df)    ~ pandas dataframes
    """
    df_cols = df.columns
    df = df.reset_index()
    nobs = df.shape[0]
    training_size = floor(nobs * train_pct)
    rand_indicies = sample(range(0, nobs), training_size)
    train_idx = df.index.isin(rand_indicies)
    train_df = df[df_cols].iloc[train_idx]
    test_df = df[df_cols].iloc[~train_idx]
    return (train_df, test_df)


###########################################################################
# Main
###########################################################################

# series_available = [
#     'Change', 'Change 1yr Treasury', 'Change 5yr Treasury',
#     'Change 10yr Treasury', 'Change 30yr Treasury',
#     'Change Implied Fed Fund', 'Change SP500', 'Change SP500 Finance'
# ]

n = 12
df_fname = DF_WITH_LDA_TEMPLATE.format(n)
lda_fname = LDA_FILE_NAME_TEMPLATE.format(n)

# load data
df = pd.read_csv(df_fname).drop("Unnamed: 0", axis=1).set_index("Date")
lda = gensim.models.LdaMulticore.load(lda_fname)

# get the topic distributions frm LDA
x = get_topic_dataframe(df["Topic Dist"]) * 100  # adjust units to percent
x.index = df.index
xc = sm.add_constant(x)

y = df["Change"].apply(classify_changes)
y.index = df.index

# merge into a single dataset for splitting
data = x.merge(y, left_index=True, right_index=True)
topic_cols = [t for t in train_df.columns if "Topic" in t]

train_df, test_df = split_train_test(data)

# split training and testing into features and target
x_train = train_df[topic_cols]
y_train = to_categorical(train_df["Change"], N_CATEGORIES)
x_test = test_df[topic_cols]
y_test = to_categorical(test_df["Change"], N_CATEGORIES)

# build CNN model
inputs = Input(shape=(12,))

output_1 = Dense(32, activation="relu")(inputs)
output_2 = Dense(64, activation="relu")(output_1)
predictions = Dense(3, activation="softmax")(output_2)

model = Model(inputs=inputs, outputs=predictions)
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=100)  # starts training

# make predictions on test set 
y_pred = model.predict(x_test)
y_pred_max = np.argmax(y_pred, axis=1)
y_pred_categorical = to_categorical(y_pred_max, N_CATEGORIES)

# check results
res = metrics.f1_score(y_test, y_pred_categorical, average='micro')