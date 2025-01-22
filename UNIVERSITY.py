#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np


# In[4]:


df = pd.read_csv("universities.csv")
df


# In[6]:


np.mean(df["SAT"])


# In[7]:


np.median(df["SAT"])


# In[8]:


np.std(df["GradRate"])


# In[9]:


df.describe()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])

