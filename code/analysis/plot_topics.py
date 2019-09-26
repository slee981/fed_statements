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
LOOPS = 1

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DF_WITH_LDA = os.path.join(DATA_DIR, '{}_topic_lda_{}_loop.csv'.format(NUM_TOPICS, LOOPS))

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

x = df['Num Topics']
xc = sm.add_constant(x)

# specify regression
y = df['Change 1yr Treasury']             # <- abs value of bps changes
model = sm.OLS(y, xc)
results = model.fit()

print(results.summary())
params = results.params
intercept, slope = params[0], params[1]
res = slope * x + intercept 

plt.plot(x,res, '-r')
plt.scatter(x,y)
plt.grid()
plt.show()