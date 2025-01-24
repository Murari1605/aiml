#!/usr/bin/env python
# coding: utf-8

# In[5]:


#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
data


# In[6]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[7]:


data.info()


# In[8]:


print(type(data))
print(data.shape)


# In[9]:


data.shape


# In[10]:


data.dtypes


# In[ ]:


# drop columns that are not needed


# In[11]:


data.describe()


# In[14]:


# drop dupplicate column and unnamed column

data1 = data.drop(['Unnamed: 0',"Temp C"], axis=1)
data1


# 

# In[16]:


data1.info()


# In[19]:


# convert the month column data type to integer data type

data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[20]:


# checking for duplicated rows in the table
# print the duplicate row
data1[data1.duplicated()]


# In[21]:


data1[data1.duplicated(keep = False)]


# In[ ]:




