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

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[12]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[14]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# Description of columns:
# 1.MPG:Mileage of the car 
# 2.HP:Horse power of the car(X1 column)
# 2.VOL:Volume of the car(X2 column)
# 3.SP:Top speed(X3 column)
# 4.WT:weight of the car(X4 column)

# In[15]:


cars.info()


# In[16]:


cars.isnull().sum()


# Observations:
# 1.There are no missing values
# 2.There are 81 observations
# 3.the data types of the columns are also relevant and valid

# In[ ]:




