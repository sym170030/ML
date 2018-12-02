#!/usr/bin/env python
# coding: utf-8

# In[13]:


# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:27:26 2018

@author: Siddharth Mudbidri
"""





import pandas as pd
import numpy as np


df_path = "C:\ASM exam\cds_spread5y_2001_2016.dta"


data = pd.io.stata.read_stata("C:\ASM exam\cds_spread5y_2001_2016.dta")
data.to_csv('my_stata_file.csv')


# In[17]:


cdsdata = pd.read_csv('my_stata_file.csv',
                  low_memory = False)
cdsdata


# In[15]:


crspdata = pd.read_csv('crsp.csv')
#d3 = pd.merge(df, df1, on='gvkey') #merging based on gvkey
crspdata


# In[18]:


cdsdata['Date'] = pd.to_datetime(cdsdata['mdate'])
cdsdata['Month']= cdsdata['Date'].dt.month
cdsdata['Year']=cdsdata['Date'].dt.year

cdsdata['quarter']='4'


# In[19]:


cdsdata


# In[20]:


cdsdata.loc[cdsdata['Month']>9,"quarter"]=4


# In[21]:


cdsdata.loc[(cdsdata['Month']>6) & (cdsdata['Month']<=9),"quarter"]=3


# In[22]:


cdsdata.loc[(cdsdata['Month']>3) & (cdsdata['Month']<=6),"quarter"]=2


# In[24]:


#cdsdata[(cdsdata['Month'])<=3,"quarter"]=1

cdsdata['gvkey'] = cdsdata['gvkey'].astype(float)
cdsdata['quarter'] = cdsdata['quarter'].astype(float)
cdsdata['Year'] = cdsdata['Year'].astype(float)


# In[27]:


cdsdata.loc[(cdsdata['Month']<=3),"quarter"]=1


# In[29]:


crspdata=crspdata.rename(columns = {'GVKEY':'gvkey'})
crspdata=crspdata.rename(columns = {'datadate':'mdate'})

crspdata
# In[31]:


crspdata


# In[32]:


crspdata['Date'] = pd.to_datetime(crspdata['mdate'])
crspdata['Month']= crspdata['Date'].dt.month
crspdata['Year']=crspdata['Date'].dt.year

crspdata['quarter']='4'


# In[33]:


crspdata['mdate'].unique()


# In[34]:


crspdata.loc[crspdata['Month']>9,"quarter"]=4
crspdata.loc[(crspdata['Month']>6) & (crspdata['Month']<=9),"quarter"]=3
crspdata.loc[(crspdata['Month']>3) & (crspdata['Month']<=6),"quarter"]=2
crspdata.loc[(crspdata['Month'])<=3,"quarter"]=1

crspdata['gvkey'] = crspdata['gvkey'].astype(float)
crspdata['quarter'] = crspdata['quarter'].astype(float)
crspdata['Year'] = crspdata['Year'].astype(float)


# In[36]:


data = pd.merge(cdsdata[1:200], crspdata[1:200], on=['gvkey', 'quarter', 'Year'])


# In[37]:


data


# In[ ]:




