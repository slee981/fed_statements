###########################################################################
# Imports
###########################################################################

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import gensim
import numpy as np
import os

###########################################################################
# Storage
###########################################################################

# specs
NUM_TOPICS = range(9, 16)

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
    if change < 0:
        return -1
    elif change == 0:
        return 0
    elif change > 0:
        return 1
    else:
        raise TypeError


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

# used for lasso
x = get_topic_dataframe(df["Topic Dist"]) * 100  # adjust units to percent
x.index = df.index
xc = sm.add_constant(x)

y = df["Change"].apply(classify_changes)
y.index = df.index

model = sm.OLS(y, xc)

lasso_res = model.fit_regularized(method="elastic_net", alpha=0.3, L1_wt=1.0)
params = lasso_res.params.drop("const")
keep_idxs = [i for i, coef in enumerate(params) if coef != 0]
keep_names = params.index[keep_idxs]

# used for restricted ols
x_restricted = x[keep_names].copy()  # <- suggested by lasso regression w/ alpha = 0.3
xc = sm.add_constant(x_restricted)

# specify regression
model = sm.OLS(y, xc)
ols_res = model.fit()
print(ols_res.summary())  # <- when you want to save, use ols_res.summary().as_latex()
