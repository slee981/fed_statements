###########################################################################
# Imports
###########################################################################

import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import os

###########################################################################
# Storage
###########################################################################

# specs 
NUM_TOPICS = 10

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DF_WITH_LDA = os.path.join(DATA_DIR, '{}_topic_lda.csv'.format(NUM_TOPICS))

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
    cols = ['Topic{}'.format(i+1) for i in range(num_topics)]
    for r in range(rows): 
        obs = str_to_arr(topic_dists.iloc[r])
        x.append(obs)
    return pd.DataFrame(x, columns=cols)

###########################################################################
# Main
###########################################################################

# series labels 
series = [
    'Change', 'Change 1yr Treasury', 'Change 5yr Treasury',
    'Change 10yr Treasury', 'Change 30yr Treasury',
    'Change Implied Fed Fund', 'Change SP500', 'Change SP500 Finance'
]

# load data 
df = pd.read_csv(DF_WITH_LDA).drop('Unnamed: 0', axis=1)

x = get_topic_dataframe(df['Topic Dist'])
xc = sm.add_constant(x)

x_restricted = x[['Topic1', 'Topic7', 'Topic9']].copy()  # <- suggested by lasso regression w/ alpha = 0.3
xc = sm.add_constant(x_restricted)

# specify regression
y = df['Change'] * 10**4
model = sm.OLS(y, xc)
ols_res = model.fit()
lasso_res = model.fit_regularized(method='elastic_net', alpha=0.3, L1_wt=1.0)
print(lasso_res.params)
print(ols_res.summary())

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