# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 12:39:08 2018

@author: Admin
"""

from urllib.request import urlopen
import zipfile
import json
import csv
import pandas as pd
import collections
import numpy as np

path = r'C:\Users\siddh\Desktop\ReportDownload\compustat_annual_2000_2017_with link information.csv'
df = pd.read_csv(path)

df2 = df.loc[:, pd.notnull(df).sum()>len(df)*.7] #remogin variables with >70% null values

df3 = df2.describe()