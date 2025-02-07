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


# In[7]:


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


# In[8]:


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


# In[9]:


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

# In[10]:


cars[cars.duplicated()]


# In[11]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[12]:


cars.corr()


# Observations:
# 1.The highest positive correlation between weight vs volume(0.999203)
# 2.the second  highest positive correlation between hp vs sp (0.973848)
# 

# In[13]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[14]:


model1.summary()


# Observations from model summary:
# 1.The R-squared and adjusted R-quared values are good and about 75% of variability in Y  is explained X columns.
# 2.The probability value with respect to F-statistic is close to zero,indicating that all or someof X columns are significant.
# 3.The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves,which need to be further explored.

# In[15]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head


# In[16]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[17]:


from sklearn.metrics import mean_squared_error
print("MSE :", mean_squared_error(df1["actual_y1"], df1["pred_y1"]))


# In[19]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse)
print("RMSE :", np.sqrt(mse))


# Checking for muliticollineraity among X-columns using VIF method

# In[20]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared
vif_wt = 1/(1-rsq_wt)

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared
vif_vol = 1/(1-rsq_vol)

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared
vif_sp = 1/(1-rsq_sp)

# Storing vif values in a data frame
d1 = {'Variables' : ['Hp' ,'WT' , 'VOL' ,'SP' ], 'VIF' : [vif_hp,vif_wt, vif_vol, vif_sp]}
Vif_frame = pd.DataFrame(d1)
Vif_frame


# Observations:
# 1.The ideal range of VIF values shall be between 0 to 10. However slightly higher values can be tolareted.
# 2.As seen from the very high VIF values for VOL and WT,it is clear that they are prone to multicollinearity proble.
# 3.Hence it is decided to drop one of the columns(either VOL or WT) to overcome the multicollinearity.
# 4.it is decided to drop WT and retain VOL column in further models

# In[21]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[22]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[23]:


model2.summary()


# In[33]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head


# In[30]:


pred_y2 = model2.predict(cars.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[31]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :", np.sqrt(mse))


# Observations from model2 summary()
# 1.The adjusted R-squared value improved slightly to 0.76.
# 2.All the p-values for model parameterrsa re lessthan 5% hence they are significant.
# 3.Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPg response variable.
# 4.there is no improvement in MSE value

# In[ ]:




