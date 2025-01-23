#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import numpy as np


# In[ ]:





# In[19]:


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


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[2]:


#visualization using boxplot


# In[7]:


#create a pandas series of batsman1 scores
s1 = [20,15,10,25,30,35,28,40,45,60]
scores1 = pd.Series(s1)
scores1


# In[10]:


plt.boxplot(scores1, vert=False)


# In[14]:


s2 = [10,5,25,45,34,56,44,150,134]
scores2 = pd.Series(s2)
scores2


# In[15]:


plt.boxplot(scores2, vert=False)


# In[17]:


df = pd.read_csv("universities.csv")
print(df)

plt.boxplot(df["SAT"])


# In[21]:


df = pd.read_csv("universities.csv")
print(df)

plt.boxplot(df["ACCEPT"])


# In[ ]:




