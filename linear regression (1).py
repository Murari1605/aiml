#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


# In[2]:


data1 =pd.read_csv("NewspaperData.csv")
data1


# In[3]:


data1.info()


# In[4]:


data1.isnull().sum()


# In[5]:


data1.describe()


# In[6]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[7]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[8]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Sunday Sales")
plt.boxplot(data1["sunday"], vert = False)
plt.show()


# In[9]:


sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# # Observations
# There are no missing values.
# The daily column values appears to be right-skewed.
# The sunday column values also appear to be right-skewed.
# There are two outliers in both daily column and also in sunday coiumn as observed from the boxplots. 

# In[10]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[11]:


data1["daily"].corr(data1["sunday"])


# In[12]:


data1[["daily","sunday"]].corr()


# observations
# 1.The relationshio between x(daily) and y(sunday) is seen to be linear as seen from scatter plot.
# 2.the correlation is strong positive wiyh person's correlation coefficient of 0.958154
# 

# In[14]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[ ]:





# In[ ]:




