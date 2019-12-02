###########################################################################
# Author       : Stephen Lee
# Date         : 10.4.19
# Goal         : Perform a leave-one-out prediction

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
NUM_TOPICS = 12

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data")
DF_WITH_LDA_TEMPLATE = os.path.join(DATA_DIR, "lda_dataframes", "{}_topic_lda.csv")
LDA_FILE_NAME_TEMPLATE = os.path.join(DATA_DIR, "lda_models", "{}_topic_lda")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

###########################################################################
# Functions
###########################################################################


def get_topic_dataframe(df):
    """ 
    INPUT
        ~ pandas df with topic distributions stored as list in the dataframe 
    OUTPUT 
        ~ pandas df with topics each in their own column 
    """

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


###########################################################################
# Main
###########################################################################

df_fname = DF_WITH_LDA_TEMPLATE.format(NUM_TOPICS)
lda_fname = LDA_FILE_NAME_TEMPLATE.format(NUM_TOPICS)

# load data
df = pd.read_csv(df_fname).drop("Unnamed: 0", axis=1)
lda = gensim.models.LdaMulticore.load(lda_fname)

# used for lasso
x = get_topic_dataframe(df["Topic Dist"])
x = sm.add_constant(x)
y = df["Change"] * 10 ** 4  # <- scale from decimal pct to bps

topic_names = []
y_actuals = []
y_preds = []
y_mis = []

# loop through and leave one out each time
# 1) record the topics selected, notify if different
# 2) record the OLS results and prediction made
# 3) average the difference between predicitive results and actual results
for idx in range(x.shape[0]):

    # TODO: specify the leave one out sample, fit, predict, store

    x_fit = x.drop(idx).copy()
    x_fit = sm.add_constant(x_fit)
    x_test = x.iloc[idx].copy().to_frame().T

    y_fit = y.drop(idx).copy()
    y_test = y.iloc[idx].copy()

    model = sm.OLS(y_fit, x_fit)
    lasso_res = model.fit_regularized(method="elastic_net", alpha=0.3, L1_wt=1.0)
    params = lasso_res.params.drop("const")
    keep_idxs = [i for i, coef in enumerate(params) if coef != 0]
    keep_names = params.index[keep_idxs]

    # used for restricted ols
    x_fit = x_fit[keep_names].copy()  # <- suggested by lasso regression w/ alpha = 0.3
    x_fit = sm.add_constant(x_fit)
    x_test = x_test[keep_names].copy()
    x_test.insert(
        0, "const", 1.0
    )  # <- force add constant. sm.add_constant doesn't work here for some reason

    # specify regression
    model = sm.OLS(y_fit, x_fit)
    ols_res = model.fit()

    # out of sample prediciton
    y_pred = ols_res.predict(x_test)
    y_pred = y_pred.values[0]  # <- comes back as pandas series

    # store info
    for topic in keep_names:
        if topic not in topic_names:
            topic_names.append(topic)
            print("Added topic {} in loop {}".format(topic, idx))
    y_actuals.append(y_test)
    y_preds.append(y_pred)
    y_mis.append(y_test - y_pred)
    print("Result: {}".format(y_test - y_pred))

# MAKE PLOTS

dates = [pd.to_datetime(d) for d in df["Date"]]

# histogram plot of errors
plt.hist(y_mis, bins=50)
plt.title("Histogram of errors (out of sample prediction)")
plt.show()

# plot the leave-one-out, out of sample predictions with the actual predictions
fname = "leave_one_out_{}_topcis.png".format(NUM_TOPICS)
fpath = os.path.join(FIGURES_DIR, fname)
t = [i for i in range(111)]
plt.plot_date(dates, y_actuals, color="blue", ms=2)
plt.plot_date(dates, y_preds, color="red", ls="-", lw=1, ms=1)
plt.title("Estimated changes vs. actual (out of sample prediction)")

save = input("Save figure? (y/n) >> ")
if "y" in save.lower():
    plt.savefig(fpath)
plt.show()


# check if errors have trend in time
plt.scatter(dates, y_mis)
plt.title("Errors across time")
plt.show()
