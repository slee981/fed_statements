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
NUM_TOPICS = 40
LOOPS = 5

# path info  - call from root
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DF_WITH_LDA = os.path.join(DATA_DIR, 'dates_{}_topic_lda_{}_loop.csv'.format(NUM_TOPICS, LOOPS))

###########################################################################
# Main
###########################################################################

# load data 
df = pd.read_csv(DF_WITH_LDA).drop('Unnamed: 0', axis=1)
y = np.abs(df['Change'] * 10**4)              # <- abs value of bps changes
x = df['Num Topics']

xc = sm.add_constant(x)
model = sm.OLS(y, xc)
results = model.fit()

print(results.summary())
intercept, slope = results.params
res = slope * x + intercept 

plt.plot(x,res, '-r')
plt.scatter(x,y)
plt.grid()
plt.show()