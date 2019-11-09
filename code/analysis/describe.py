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


###########################################################################
# Main
###########################################################################

df_fname = DF_WITH_LDA_TEMPLATE.format(NUM_TOPICS)
lda_fname = LDA_FILE_NAME_TEMPLATE.format(NUM_TOPICS)

# load data
df = pd.read_csv(df_fname).drop("Unnamed: 0", axis=1)
lda = gensim.models.LdaMulticore.load(lda_fname)

# df columns
# Index(['index', 'Actual', 'Previous', 'Change', 'Date', '2-gram',
#        'Speech Dates', 'Stemmed', 'Change 1yr Treasury', 'Change 5yr Treasury',
#        'Change 10yr Treasury', 'Change 30yr Treasury',
#        'Change Implied Fed Fund', 'Change SP500', 'Change SP500 Finance',
#        'Topic Dist'],
#       dtype='object')

# variables of interest
x = get_topic_dataframe(df["Topic Dist"])
y = df["Change"] * 10 ** 4
# df = x.merge(y.to_frame(), left_index=True, right_index=True)

# PLOT: ACTUAL INTEREST RATES and topics over time

fname = os.path.join(ROOT_DIR, "figures", "topics_panel.png")
fig, (ax, ax1, ax2, ax3, ax4) = plt.subplots(5, sharex=True)

# plot estimated against actual
dates = [pd.to_datetime(d) for d in df["Date"]]
ax.plot_date(dates, y, color="blue", ls="-", lw=2, ms=1)
ax.set_title("Change in Federal Funds Rate (FFR)")
# ax.set(ylabel="Change in Federal Funds Interest Rate (bps)")

# PLOT 2: TOPIC 3
topic_three = x["Topic3"]

# plot estimated against actual
dates = [pd.to_datetime(d) for d in df["Date"]]
ax1.plot_date(dates, topic_three, color="blue", ls="-", lw=2, ms=1)
ax1.set_title("Pct. Discussing Topic 3")
# ax1.set(ylabel="Pct. Discussion about Topic 3")

# topic 7
topic_seven = x["Topic7"]

# plot estimated against actual
dates = [pd.to_datetime(d) for d in df["Date"]]
ax2.plot_date(dates, topic_seven, color="blue", ls="-", lw=2, ms=1)
ax2.set_title("Pct. Discussing Topic 7")
# ax2.set(ylabel="Pct. Discussion about Topic 7")


# topic 10
topic_ten = x["Topic10"]

# plot estimated against actual
dates = [pd.to_datetime(d) for d in df["Date"]]
ax3.plot_date(dates, topic_ten, color="blue", ls="-", lw=2, ms=1)
ax3.set_title("Pct. Discussing Topic 10")
# ax3.set(xlabel="Date", ylabel="Pct. Discussion about Topic 10")

# topic 11
topic_eleven = x["Topic11"]

# plot estimated against actual
dates = [pd.to_datetime(d) for d in df["Date"]]
ax4.plot_date(dates, topic_eleven, color="blue", ls="-", lw=2, ms=1)
ax4.set_title("Pct. Discussing Topic 11")
# ax4.set(xlabel="Date", ylabel="Pct. Discussion about Topic 11")

# SAVE PLOT
fig.tight_layout()
fig.savefig(fname)


