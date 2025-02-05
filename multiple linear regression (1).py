#!/usr/bin/env python
# coding: utf-8

# Assumptions in Multilinear Regression
# 
# 1. Linearity: The relationship between the predictors(X)     and the response(Y) is linear.
# 
# 2. Independence: Observations are independent of each       other.
# 
# 3. Homoscedasticity: The residuals (Y - Y_hat)) exhibit     constant variance at all levels of the predictor.
# 
# 4. Normal Distribution of Errors: The residuals of the       model are normally distributed.
# 
# 5. No multicollinearity: The independent variables should   not be too highly correlated with each other.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# Description of columns:
# 1.MPG:Mileage of the car (Y-Column)
# 2.HP:Horse power of the car(X1-Column)
# 2.VOL:Volume of the car(X2-Column)
# 3.SP:Top speed(X3-Column)
# 4.WT:weight of the car(X4-Column)

# In[4]:


cars.info()


# In[5]:


cars.isnull().sum()


# Observations:
# 1.There are no missing values
# 2.There are 81 observations
# 3.the data types of the columns are also relevant and valid

# In[6]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust Layout
plt.tight_layout()
plt. show()


# In[8]:


#Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist)=plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})
# Create a boxplot
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='') #Remove x label for the boxplot 
#Create a histogram in the same x-axis
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()


# In[9]:


#Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist)=plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})
# Create a boxplot
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='') #Remove x label for the boxplot 
#Create a histogram in the same x-axis
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()


# In[10]:


#Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist)=plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})
# Create a boxplot
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='') #Remove x label for the boxplot 
#Create a histogram in the same x-axis
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()


# Observatins from boxplot and histograms:
# 1.There are some values obsorved in towards the right tail SP and HP distributions.
# 2.In vol and wt columns,a few outliers are obsorved in both tails of their distributors.
# 3.The extreme values of cars data may have come from the specially designed nature of cars.

# In[11]:


cars[cars.duplicated()]


# In[12]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[13]:


cars.corr()


# Observations:
# 1.The highest positive correlation between weight vs volume(0.999203)
# 2.the second  highest positive correlation between hp vs sp (0.973848)
# 

# In[16]:


model = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[17]:


model.summary()


# In[ ]:




