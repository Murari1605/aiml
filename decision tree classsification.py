#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as  np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[3]:


data = pd.read_csv("iris.csv")
data


# In[9]:


import seaborn as sns
data["variety"].value_counts()


# In[10]:


data.info()


# In[11]:


data[data.duplicated(keep= False)]


# # Observations:
#  - There are 150 rows and 5 columns
#  -  There are no null values
#  - there is one duplicated row
#  - The x-columns are sepal.length,sepal.width,petal.length and petal.width
#  - All the x-columns are continious
#  - The y-column is "variety" which is categorical
#  - There are three flower categories (classes)

# In[13]:


data = data.drop_duplicates(keep='first')


# In[14]:


data[data.duplicated]


# In[17]:


data = data.reset_index(drop= True)
data


# In[ ]:




