# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:55:53 2018

@author: Admin
"""

import pandas as pd
import numpy as np

df_path = "C:\ASM exam\cds_spread5y_2001_2016.dta"


data = pd.io.stata.read_stata("C:\ASM exam\cds_spread5y_2001_2016.dta")
data.to_csv('my_stata_file.csv')

df = pd.read_csv('my_stata_file.csv')
print(df['gvkey'])
a = df.gvkey.unique()
np.savetxt('k1.txt', a,fmt='% 4d') ##saving gvkeys into text file

df1 = pd.read_csv('crsp.csv')
d3 = pd.merge(df, df1, on='gvkey') #merging based on gvkey
