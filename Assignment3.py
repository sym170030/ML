# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 00:09:20 2018

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

from sklearn.tree import DecisionTreeClassifier  

import statsmodels.api as sm

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

Y = df6['oibdp']
df6 = df6.drop(columns=['oibdp'])
X = df6

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)  
regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test)  

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  






