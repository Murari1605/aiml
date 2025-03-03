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


# In[2]:


data = pd.read_csv("iris.csv")
data


# In[3]:


import seaborn as sns
data["variety"].value_counts()


# In[4]:


data.info()


# In[5]:


data[data.duplicated(keep= False)]


# # Observations:
#  - There are 150 rows and 5 columns
#  -  There are no null values
#  - there is one duplicated row
#  - The x-columns are sepal.length,sepal.width,petal.length and petal.width
#  - All the x-columns are continious
#  - The y-column is "variety" which is categorical
#  - There are three flower categories (classes)

# In[6]:


data = data.drop_duplicates(keep='first')


# In[7]:


data[data.duplicated]


# In[8]:


data = data.reset_index(drop= True)
data


# In[9]:


labelencoder = LabelEncoder()
data.iloc[:,-1] = labelencoder.fit_transform(data.iloc[:,-1])
data.head()


# In[10]:


data.info()


# In[11]:


data['variety'] = pd.to_numeric(labelencoder.fit_transform(data['variety']))
print(data.info())


# In[12]:


X=data.iloc[:,0:4]
Y=data['variety']


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state = 1)
x_train


# In[16]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth =None)
model.fit(x_train,y_train)


# In[17]:


plt.figure(dpi=1200)
tree.plot_tree(model);


# In[19]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names = fn, class_names=cn, filled = True);


# In[ ]:




