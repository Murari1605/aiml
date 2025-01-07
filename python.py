#!/usr/bin/env python
# coding: utf-8

# In[1]:


L = [100,53,200,79,20,89,1007]


# In[3]:


even =[]
for x in L:
    if x%2 == 0:
        even.append(x)
print(even)   


# In[4]:


odd =[]
for x in L:
    if x%2 == 1:
        odd.append(x)
print(odd)   


# In[7]:


order = [['apple',10], ['banana',5],['cherry',7],['dragon',10],['grapes',9],['apple',20]]
for each in order:
    if each[0] == "cherry":
        each[1] = 10
print(order)    


# In[9]:


order = [['apple',10], ['banana',5],['cherry',7],['dragon',10],['grapes',9],['apple',20]]
total = []
for each_item in orders:
    if each_item[0] == "apple":
        total.append(each_item[1])
print(order)
print(total)
print(sum(total))


# In[12]:


tup1 = (4,10,9,"A",9.81,False,9-6j)
print (tup1)
print(type(tup1))


# In[13]:


tup1 = (4,10,9,"A",9.81,False,9-6j)
tup1[4:6]


# In[14]:


tup1 = (4,10,9,"A",9.81,False,9-6j)
tup1.index(4)


# In[15]:


tup1 = (4,10,9,"A",9.81,False,9-6j,4,"B",4,4,4,5,6,1)
tup1.count(4)


# In[1]:


list1 = [1,8,9,0,10,20,78,8,8,8]


# In[2]:


s2 = set(list1)
print(s2)
print(type(s2))


# In[3]:


s1 = {1,2,3,4}
s2 = {3,4,5,6}


# In[4]:


s1 & s2


# In[5]:


s1 - s2


# In[6]:


s2 - s1


# In[7]:


s1 = {1,2,3,4}
s2 = {2,5,7,3}
s1.symmetric_difference(s2)


# In[8]:


str1 = 'murari'
print(str1)
str1[2:5]


# In[12]:


sales_data = {
    "ProductID": [101,102,103,104],
    "ProductName": ["chair","keyboard","mouse"],
    "Category": ["Furniture"]
    
}
for k,v in sales_data.items():
    print(k,set(v))
    print('/n')


# In[ ]:




