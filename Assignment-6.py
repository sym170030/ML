#!/usr/bin/env python
# coding: utf-8

# In[51]:


# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:27:26 2018

@author: Siddharth Mudbidri
"""





import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import  mean_squared_error
import numpy as np


df_path = "C:\ASM exam\cds_spread5y_2001_2016.dta"


data = pd.io.stata.read_stata("C:\ASM exam\cds_spread5y_2001_2016.dta")
data.to_csv('my_stata_file.csv')


# In[52]:


cdsdata = pd.read_csv('my_stata_file.csv',
                  low_memory = False)
cdsdata


# In[53]:


crspdata = pd.read_csv('crsp.csv')
#d3 = pd.merge(df, df1, on='gvkey') #merging based on gvkey
crspdata


# In[54]:


cdsdata['Date'] = pd.to_datetime(cdsdata['mdate'])
cdsdata['Month']= cdsdata['Date'].dt.month
cdsdata['Year']=cdsdata['Date'].dt.year

cdsdata['quarter']='4'


# In[55]:


cdsdata


# In[56]:


cdsdata.loc[cdsdata['Month']>9,"quarter"]=4


# In[57]:


cdsdata.loc[(cdsdata['Month']>6) & (cdsdata['Month']<=9),"quarter"]=3


# In[58]:


cdsdata.loc[(cdsdata['Month']>3) & (cdsdata['Month']<=6),"quarter"]=2


# In[59]:


#cdsdata[(cdsdata['Month'])<=3,"quarter"]=1

cdsdata['gvkey'] = cdsdata['gvkey'].astype(float)
cdsdata['quarter'] = cdsdata['quarter'].astype(float)
cdsdata['Year'] = cdsdata['Year'].astype(float)


# In[60]:


cdsdata.loc[(cdsdata['Month']<=3),"quarter"]=1


# In[61]:


crspdata=crspdata.rename(columns = {'GVKEY':'gvkey'})
crspdata=crspdata.rename(columns = {'datadate':'mdate'})

crspdata
# In[62]:


crspdata


# In[76]:


crspdata['Date'] = pd.to_datetime(crspdata['mdate'])
crspdata['Month']= crspdata['Date'].dt.month
crspdata['Year']=crspdata['Date'].dt.year

crspdata['quarter']='4'


# In[77]:


crspdata['mdate'].unique()


# In[65]:


crspdata.loc[crspdata['Month']>9,"quarter"]=4
crspdata.loc[(crspdata['Month']>6) & (crspdata['Month']<=9),"quarter"]=3
crspdata.loc[(crspdata['Month']>3) & (crspdata['Month']<=6),"quarter"]=2
crspdata.loc[(crspdata['Month'])<=3,"quarter"]=1

crspdata['gvkey'] = crspdata['gvkey'].astype(float)
crspdata['quarter'] = crspdata['quarter'].astype(float)
crspdata['Year'] = crspdata['Year'].astype(float)


# In[73]:


merge = pd.merge(cdsdata[1:200], crspdata[1:200], on=['gvkey', 'quarter', 'Year'])


# In[78]:


crspdata


# In[2]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
merge = merge.select_dtypes(include=numerics)
merge = merge.fillna(data.median())
merge = merge.dropna(axis=1, how='any')

#Split the dataset 
TestData = merge[(merge['Year'] >= 2010) & (merge['Year'] <= 2018)]

#Initialze X and Y
Xt= TestData.drop('spread5y', axis=1)
yt=TestData['spread5y']

Xt= Xt.drop('Month_x', axis=1)
Xt= Xt.drop('Month_y', axis=1)
Xt= Xt.drop('quarter', axis=1)
Xt= Xt.drop('Year', axis=1)
Xt= Xt.drop('gvkey', axis=1)

TrainData=data[(data['Year'] < 2011)]

#splitting x and y for test data
X_train= TrainData.drop('spread5y', axis=1)
y_train=TrainData['spread5y']
X_train= X_train.drop('Month_x', axis=1)
X_train= X_train.drop('Month_y', axis=1)
X_train= X_train.drop('quarter', axis=1)
X_train= X_train.drop('Year', axis=1)
X_train= X_train.drop('gvkey', axis=1)


# In[3]:


randomforest = RandomForestRegressor(n_estimators=100) 
randomforest.fit(X_train, y_train)
randomforest_Predictor=randomforest.predict(X_test)


# In[4]:


X_train1F=X_train[newfeatures]
X_test1F=X_test[newfeatures]


scaler = StandardScaler()
scaler.fit(X_train1)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train_NewFt=scaler.transform(X_train1)
X_test_NewFt=scaler.transform(X_test1)

np.random.seed(10)
classifier = Sequential()
classifier.add(Dense(32, input_dim=50, activation='relu'))
classifier.add(Dense(8, activation='relu'))
classifier.add(Dense(1, activation='relu')) 
classifier.compile(optimizer='adam',loss='mse',  metrics=['accuracy'])
classifier.fit(X_train_NewFt, y_train, batch_size=15 epochs=100,)


# In[5]:


Y_pred = model.predict(X_test_NewFt)

mape=mean_absolute_percentage_error(y_test,Y_pred)

mape


# In[6]:





# In[7]:





# In[ ]:




