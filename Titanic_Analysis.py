#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[3]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[4]:


titanic.info()


# Observations:
# 1.There is no null values.
# 2.All columns are object type and categorical in nature.
# 3.As the columns are categorical,we can adopt one-hot-encoding.

# In[5]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[6]:


counts = titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# In[7]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# In[8]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# In[9]:


df = pd.get_dummies(titanic,dtype=int)
df.head()


# In[10]:


df.info()


# Aprior Algorithm

# In[11]:


frequent_itemsets = apriori(df, min_support = 0.05, use_colnames=True, max_len=None)
frequent_itemsets


# In[13]:


frequent_itemsets.info()


# In[14]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# In[16]:


rules.sort_values(by='lift', ascending = False)


# In[17]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




