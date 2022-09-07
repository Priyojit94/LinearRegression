#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,r2_score
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv('F:/LineraRegression assin/Salary_Data.csv')
data.head()


# In[4]:


data.shape


# In[5]:


data.isnull().sum().sum()


# In[6]:


data.dtypes


# In[14]:


corr=data.corr()
corr


# In[16]:


sns.heatmap(corr,annot=True)


# In[7]:


sns.regplot(x='YearsExperience',y='Salary',data=data,color='green')


# In[12]:


plt.figure(figsize=(20,15))
sns.barplot('YearsExperience','Salary',data=data)


# In[23]:


sns.displot(x=data['Salary'])


# In[24]:


sns.displot(x=data['YearsExperience'])


# In[28]:


x=data.iloc[:,0].values
x


# In[29]:


x=x.reshape(-1,1)
x


# In[30]:


y=data.iloc[:,1].values
y=y.reshape(-1,1)
y


# In[31]:


model=LinearRegression()
model_fitting=model.fit(x,y)


# In[32]:


model_fitting.coef_


# In[33]:


model_fitting.intercept_


# In[37]:


accuracy=model_fitting.score(x,y)
accuracy


# In[38]:


pred=model_fitting.predict(x)
pred


# In[57]:


residuals=pd.Series({'Actual':y,'predict':pred,'Error':(y-pred)})
residuals


# In[69]:


import numpy as np


# In[73]:


Year_experience=np.array([11,11.5,12])
Year_experience


# In[74]:


Year_experience=Year_experience.reshape(-1,1)
Year_experience


# In[75]:


new=model_fitting.predict(Year_experience)
new


# In[ ]:




