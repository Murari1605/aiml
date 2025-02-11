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


# In[3]:


Univ.info()


# In[4]:


Univ.corr()


# In[5]:


Univ.describe()


# #### Standardization of a data

# In[6]:


Univ1 = Univ.iloc[:,1:]


# In[7]:


Univ1


# In[8]:


cols = Univ1.columns


# In[9]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1), columns = cols )
scaled_Univ_df


# In[19]:


from  sklearn.cluster import KMeans
cluster_new = KMeans(3, random_state=0)
cluster_new.fit(scaled_Univ_df)


# In[22]:


cluster_new.labels_


# In[23]:


set(cluster_new.labels_)


# In[24]:


Univ['clusterid_new'] = cluster_new.labels_


# In[25]:


Univ


# In[26]:


Univ.sort_values(by = "clusterid_new")


# In[27]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# Observations:
# 1.Cluster 2 appears to be the top most universities cluster as the cut off score.Top 10,SFRatio parameter mean values are highest.
# 2.Cluster 1 appears to occupy the midddle level rated universities.
# 3.Cluster 0 comes as the lower level rated universities.

# In[28]:


Univ[Univ['clusterid_new']==0]


# In[ ]:




