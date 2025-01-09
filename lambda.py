#!/usr/bin/env python
# coding: utf-8

# In[1]:


def mean_value(given_list):
    total = sum(given_list)
    average_value = total/len(given_list)
    return average_value


# In[6]:


def greet(name):
    print(f"Good Morning {name}!")


# In[7]:


greet("Murari")


# In[1]:


greet = lambda name : print(f"Good Morning {name}!")


# In[2]:


greet("Murari")


# In[6]:


product = lambda a,b,c : a*b*c


# In[7]:


product(20,30,40)


# In[10]:


even = lambda L : [x for x in L if x%2 ==0]


# In[11]:


my_list = [100,3,9,38,43,56,20]
even(my_list)


# In[12]:


odd = lambda L : [x for x in L if x%2 ==1]


# In[13]:


my_list = [100,3,9,38,43,56,20]
odd(my_list)


# In[19]:





# In[20]:


def product(*n):
    result=1
    for i in range(len(n)):
        result *= n[i]
    return result    


# In[18]:


product(5,4,9)


# In[21]:


def mean_value(*n):
    sum = 0
    counter = 0
    for x in n:
        counter = counter +1
        sum += x
    mean = sum/counter
    return mean


# In[22]:


mean_value(3,4,5,6,7,8,1,2)


# In[ ]:




