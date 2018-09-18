# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 12:39:08 2018

@author: Admin
"""

from urllib.request import urlopen
from sklearn.datasets import load_boston
import zipfile
import json
import csv
import pandas as pd
import collections
import numpy as np

import statsmodels.api as sm

###################################################

#data = load_boston()
#X = pd.DataFrame(data.data, columns=data.feature_names)
#y = data.target


def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    
    
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included



path = r'C:\Users\siddh\Desktop\ReportDownload\compustat_annual_2000_2017_with link information.csv'
df = pd.read_csv(path)

df2 = df.loc[:, pd.notnull(df).sum()<len(df)*.7] #removing variables with >70% null values

df3 = df2.select_dtypes(['number'])

# =============================================================================
# df
# =============================================================================

df4 = df3.fillna(df.median())

df6 = df[df.columns[df.isnull().sum()/df.shape[0]<0.7]]
df6.shape

df6 = df6.select_dtypes(include=np.number)
df6
X = df6

#cols = [col for col in df6.columns if 'ebit' in col]
#print(list(df6.columns))
#print(cols)

#
y =  df6["ebit"].tolist()
result = stepwise_selection(X, y)
result
#print(result)

















####################################################
