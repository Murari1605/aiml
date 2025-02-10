#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities1.csv")
Univ


# In[6]:


Univ.info()


# In[7]:


Univ.corr()


# In[9]:


Univ.describe()


# #### Standardization of a data

# In[10]:


Univ1 = Univ.iloc[:,1:]


# In[12]:


Univ1


# In[13]:


cols = Univ1.columns


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1), columns = cols )
scaled_Univ_df


# In[ ]:




