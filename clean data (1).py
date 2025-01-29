#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
data


# In[3]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[4]:


data.info()


# In[5]:


print(type(data))
print(data.shape)


# In[6]:


data.shape


# In[7]:


data.dtypes


# In[8]:


# drop columns that are not needed


# In[9]:


data.describe()


# In[10]:


# drop dupplicate column and unnamed column

data1 = data.drop(['Unnamed: 0',"Temp C"], axis=1)
data1


# 

# In[11]:


data1.info()


# In[12]:


# convert the month column data type to integer data type

data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[13]:


# checking for duplicated rows in the table
# print the duplicate row
data1[data1.duplicated()]


# In[14]:


data1[data1.duplicated(keep = False)]


# In[15]:


data1.rename({'solar': 'solar'}, axis=1, inplace = True)
data1


# In[16]:


data1.isnull().sum()


# In[17]:


cols = data1.columns
colours = ['black', 'yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[18]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[19]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[20]:


median_Solar = data1["Solar"].median()
mean_Solar = data1["Solar"].mean()
print("Median of Solar: ", median_Solar)
print("Mean of Ozone: ", mean_Solar)


# In[ ]:


data1['Solar.R'] = data1['Solar.R'].fillna(median_Solar.R)
data1.isnull().sum()


# In[ ]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[ ]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[ ]:


data1["Month"] = data1["Month"].fillna(mode_weather)
data1.isnull().sum()


# In[ ]:


data1.tail()


# In[ ]:


data1.reset_index(drop=True)


# In[ ]:


#Create a figure with two subplots ,stacked vertically
fig, axes=plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1, 3]})
#Plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Ozone"], ax=axes[0],color='skyblue',width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

#Plot the histogram with KDE Curve in the second (bottom) subplot
sns.histplot(data1["Ozone"], kde=True, ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
 
#Adjust layout for better spacing
plt.tight_layout()

#Show the plot
plt.show()




# # observations
# .the ozone colun has extreme values beyond 81 as seen from box plot
# .the same is confirmed from the below right-skewed histogram

# In[ ]:


#Create a figure with two subplots ,stacked vertically
fig, axes=plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1, 3]})
#Plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Solar"], ax=axes[0],color='skyblue',width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")

#Plot the histogram with KDE Curve in the second (bottom) subplot
sns.histplot(data1["Solar"], kde=True, ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("solar Levels")
axes[1].set_ylabel("Frequency")
 
#Adjust layout for better spacing
plt.tight_layout()

#Show the plot
plt.show()


# In[22]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert=False)


# In[23]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[24]:


# Method 2 for outlier detection
data1["Ozone"].describe()


# In[25]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x < (mu -3*sigma)) or (x > (mu + 3*sigma))): 
        print(x)


# In[26]:


import scipy.stats as stats
plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# In[27]:


sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("Violin Plot")
plt.show()


# In[31]:


sns.swarmplot(data=data1, x = "Weather", y = "Ozone",color="orange",palette="Set2", size=6)


# In[38]:


sns.stripplot(data=data1, x = "Weather", y = "Ozone",color="orange",palette="Set1", size=6, jitter = True)


# In[37]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"], color="black")


# In[39]:


sns.boxplot(data = data1, x = "Weather", y="Ozone")


# In[40]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[43]:


data1["Wind"].corr(data1["Temp"])


# we can observe the data it have mild negative correlation between two variables

# In[ ]:




