#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


insulation = np.array([3, 3, 10, 6, 6, 6, 6, 10, 10, 3, 10, 6, 3, 3, 10]).reshape(-1,1)
temp = np.array([40, 27, 40, 73, 64, 34, 9, 8, 23, 63, 65, 41, 21, 38, 58]).reshape(-1,1)
oil_gal = np.array([275.3, 363.8, 164.3, 40.80, 94.3, 230.9, 366.7, 300.6, 237.8, 121.4, 31.4, 203.5, 441.1, 323.0, 52.5]).reshape(-1,1)
insulation_and_temp=np.array([[40,3],[27,3], [40,10],[ 73,6],[ 64,6],[ 34,6],[ 9,6],[ 8,10],[ 23,10], [63,3],[ 65,10],[ 41,6],[ 21,3],[ 38,3],[ 58,10]]).reshape(-1,2)


# ### quiz 1

# In[21]:


a = linear_model.LinearRegression();
a.fit(insulation, oil_gal);

b = linear_model.LinearRegression();
b.fit(temp, oil_gal);

c = linear_model.LinearRegression();
c.fit(insulation_and_temp, oil_gal);


# In[24]:


print("in model a")
print(f'b1 = {a.coef_[0]}')
print(f'b0 = {a.intercept_}')
print(" ")
print("in model b")
print(f'b1 = {b.coef_[0]}')
print(f'b0 = {b.intercept_}')
print(" ")
print("in model c")
print(f'b1, b2= {c.coef_[0]}')
print(f'b0 = {c.intercept_}')


# ### quiz 2

# In[11]:


print(a.predict([[6]]));


# In[12]:


print(b.predict([[30]]));


# In[13]:


print(c.predict([[6, 30]]));


# ### quiz 3

# In[16]:


print("for a: R^2: {:.4f}".format(a.score(insulation, oil_gal)));
print("for b: R^2: {:.4f}".format(b.score(temp, oil_gal)));


# In[17]:


print("C: R^2: {:.4f}".format(c.score(insulation_and_temp, oil_gal)))
cr2 = c.score(insulation_and_temp, oil_gal)


# In[18]:


answer = 1 - ((1 - cr2) * ((insulation_and_temp.shape[0] - 1) / (insulation_and_temp.shape[0] - insulation_and_temp.shape[1] - 1)))
answer


# ### quiz 4

# In[25]:


def adjusted(data, r):
    return 1 - ((1 - r) * (len(data) - 1) / (len(data) - len(data[0]) - 1))

insulation2 = insulation ** 2

data = []

for i in range(len(insulation)):
    data.append([insulation[i][0], insulation2[i][0]]) 
    
b2 = linear_model.LinearRegression()
b2.fit(data, oil_gal)
b2_r = b2.score(data, oil_gal)

print(adjusted(data, b2_r))


# In[ ]:




