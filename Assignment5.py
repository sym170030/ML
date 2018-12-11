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


from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import  mean_squared_error
import numpy as np

randforst = RandomForestRegressor(n_estimators=50) 
randforst.fit(X_train, y_train)
#randforst.score(X_test, y_test)
randforst_Pred=randforst.predict(Xt)
F_imp=randforst.feature_importances_
F_imp = pd.DataFrame(randforst.feature_importances_,index = X_train.columns,columns=['imp']).sort_values('imp',ascending=False)
newfeatures=F_imp.iloc[:50,:]
newfeatures=newfeatures.index.tolist()

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_squared_error(yt,randforst_Pred)
mean_absolute_percentage_error(yt, randforst_Pred)

X_train_NewF=X_train[newfeatures]
X_test_NewF=X_test[newfeatures]


scaler = StandardScaler()
scaler.fit(X_train_NewF)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train_NewFt=scaler.transform(X_train_NewF)
X_test_NewFt=scaler.transform(X_test_NewF)


# In[4]:


newtrain_x = X_train[list(newfeatures)]
newtest_x = X_test[list(newfeatures)]

regressor_100 = RandomForestRegressor(n_estimators = 100,max_depth = 3)
regressor_100.fit(newtrain_x,y_train)
pred_100 = regressor_100.predict(newtest_x)
print('Mean Accuracy at 100:', regressor_100.score(newtest_x,y_test))
errors_100 = abs(pred_100 - y_test)
mape_100 = 100 * (errors_100 / y_test)
accuracy_100 = 100 - np.mean(mape_100)
print('Accuracy_100:', round(accuracy_100, 2), '%.')


# In[5]:


regressor_500 = RandomForestRegressor(n_estimators = 500,max_depth = 3)
regressor_500.fit(newtrain_x,y_train)
pred_500 = regressor_500.predict(newtest_x)
print('Mean Accuracy_500:', regressor_500.score(newtest_x,y_test))
errors_500 = abs(pred_500 - y_test)
mape_500 = 500 * (errors_500 / y_test)
# Calculate and display accuracy
accuracy_500 = 500 - np.mean(mape_500)
print('Accuracy_500:', round(accuracy_500, 2), '%.')


# In[6]:


from sklearn.ensemble import GradientBoostingRegressor 
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import xgboost


# In[7]:


GB_100 = ensemble.GradientBoostingRegressor(n_estimators = 100, max_depth = 3)
GB_100.fit(newtrain_x, y_train)
mse_100 = mean_squared_error(y_test, GB_100.predict(newtest_x))
print("MSE_100: %.4f" % mse_100)


# In[ ]:


GB_200 = ensemble.GradientBoostingRegressor(n_estimators = 200, max_depth = 3)
GB_200.fit(newtrain_x, y_train)
mse_200 = mean_squared_error(y_test, GB_200.predict(newtest_x))
print("MSE_200: %.4f" % mse_200)

