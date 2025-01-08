#!/usr/bin/env python
# coding: utf-8

# In[1]:


num = 6
if num%2 == 0:
    print("even")
else:
    print("odd")


# In[2]:


num = 0
result = "Positive" if num > 0 else ("Negative" if num < 0 else "zero")
print(result)


# In[3]:


L =[1,9,2,10,56,89]
[2*x for x in L]


# In[4]:


L =[1,9,2,10,56,89]
[x for x in L if x%2 == 0]


# In[6]:


#print avg value numbers of the list
L =[1,9,2,10,56,89]
sum([x for x in L])/len(L)


# In[12]:


d1 = {'Ram':[70,71,98,100], 'John': [56,98,67,65]}
{k:sum(v)/len(v) for k,v in d1.items()} 


# In[ ]:




