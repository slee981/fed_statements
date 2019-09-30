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
NUM_TOPICS = range(9,21)

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DF_WITH_LDA_TEMPLATE = os.path.join(DATA_DIR, '{}_topic_lda.csv')
LDA_FILE_NAME_TEMPLATE = os.path.join(DATA_DIR, 'models', '{}_topic_lda')

###########################################################################
# Functions
###########################################################################

def get_topic_dataframe(df):
    def str_to_arr(str_arr):
        return [float(ele) for ele in str_arr.replace('[','').replace(']','').split(',')]
    topic_dists = df.copy()
    rows = df.shape[0]
    num_topics = len(str_to_arr(topic_dists.iloc[0]))
    x = []
    cols = ['Topic{}'.format(i) for i in range(num_topics)]
    for r in range(rows): 
        obs = str_to_arr(topic_dists.iloc[r])
        x.append(obs)
    return pd.DataFrame(x, columns=cols)

###########################################################################
# Main
###########################################################################

# series_available = [
#     'Change', 'Change 1yr Treasury', 'Change 5yr Treasury',
#     'Change 10yr Treasury', 'Change 30yr Treasury',
#     'Change Implied Fed Fund', 'Change SP500', 'Change SP500 Finance'
# ]

for n in NUM_TOPICS: 
    df_fname = DF_WITH_LDA_TEMPLATE.format(n)
    lda_fname = LDA_FILE_NAME_TEMPLATE.format(n)

    # load data 
    df = pd.read_csv(df_fname).drop('Unnamed: 0', axis=1)
    lda = gensim.models.LdaMulticore.load(lda_fname)

    # used for lasso 
    x = get_topic_dataframe(df['Topic Dist'])
    xc = sm.add_constant(x)

    y = df['Change'] * 10**4
    model = sm.OLS(y, xc)
    lasso_res = model.fit_regularized(method='elastic_net', alpha=0.3, L1_wt=1.0)
    params = lasso_res.params.drop('const')
    keep_idxs = [i for i, coef in enumerate(params) if coef != 0]
    keep_names = params.index[keep_idxs]

    # used for restricted ols
    x_restricted = x[keep_names].copy()  # <- suggested by lasso regression w/ alpha = 0.3
    xc = sm.add_constant(x_restricted)

    # specify regression
    model = sm.OLS(y, xc)
    ols_res = model.fit()
    print('\n\n############# OLS with {} LDA topics ############'.format(n), end='\n')
    print(ols_res.summary())
    for i in keep_idxs: 
        print('\nTopic {} : {}'.format(i, lda.show_topic(i)), end='\n\n')

'''
x = get_dataframe(df['Change'])
y = df['Change 1yr Treasury']
xc = sm.add_constant(x)

model = sm.OLS(y, xc)
ols_res = model.fit()
params = ols_res.params
intercept, slope = params[0], params[1]
y_hat = slope * x + intercept 

plt.plot(x,y_hat, '-r')
plt.scatter(x,y)
plt.grid()
plt.show()
'''