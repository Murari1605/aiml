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

# In[13]:


# Build regression model

import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[14]:


model1.summary()


# In[15]:


# Plot the scatter plot and overlay the fitted straight line using matplotlib
x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 = 1.33
# predicted response vector
y_hat = b0 + b1*x
# plotting the regression line
plt.plot(x, y_hat, color = "g")

plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[16]:


sns.regplot(x="daily",y="sunday", data=data1)
plt.xlim([0,1250])
plt.show()


# In[17]:


#Predict sunday sales for 200 and 300 and 1500 daily circulation
newdata=pd.Series([200,300,1500])


# In[18]:


data_pred=pd.DataFrame(newdata, columns=['daily'])
data_pred


# In[19]:


model1.predict(data_pred)


# In[20]:


#predict on alla given training data

pred = model1.predict(data1["daily"])
pred


# In[21]:


data1["Y_hat"] = pred
data1


# In[22]:


# Compute the error values (residuals) and add as another column
data1["residuals"]= data1["sunday"]-data1["Y_hat"]
data1


# In[23]:


#compute mean squared error for the model
mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[24]:


#compute mean absolute error (MAE)
mae = np.mean(np.abs(data1["daily"]-data1["Y_hat"]))
mae


# In[25]:


plt.scatter(data1["Y_hat"], data1["residuals"])


# checking the model resiuals  of scatter plot and normality
# 1.check the scatter plot for residuals Vs Y_hat
# 2.check the  Q-Q plot for residuals distribution
# 3.check the kde plot for residuals

# Observations:
# 1.There appears to be no trnd and the residuals are         randomly placed around the zero error line
# 2.Hence the assupmtion of homoscedasticty(constant           variance in residuals) is satisfied

# In[26]:


# Plot the Q-Q plot (to check the normality of residuals)
import statsmodels.api as sm
sm.qqplot(data1["residuals"], line='45', fit=True)
plt.show()


# In[28]:


# plot the kde distribution for residuals
sns.histplot(data1["residuals"],kde=True)


# Observations:
# 1.The data points are seen to closely follow the             reference line of normality
# 2.Hence the residuals are approximately normally             distributr as also can be seen from the kde distribution
# 3.All the assumptions of simple linear regression are       verified and found to be valid in this case

# In[ ]:




