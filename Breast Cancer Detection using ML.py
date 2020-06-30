#!/usr/bin/env python
# coding: utf-8

# ## Breast Cancer Detection 

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#load dataset
df = pd.read_csv('data.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.isna().sum()


# In[5]:


df= df.dropna(axis=1)


# In[6]:


df.info()


# In[7]:


#count of malignant and benignant
df['diagnosis'].value_counts()


# In[8]:


sns.countplot(df['diagnosis'], label = "count")


# In[9]:


df.dtypes


# In[10]:


#encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.iloc[:, 1]= le.fit_transform(df.iloc[:,1].values)


# In[11]:


print(df.iloc[:, 1])


# In[12]:


df.head()


# In[13]:


df.corr()


# In[14]:


#heatmap
plt.figure(figsize= (20,20))
sns.heatmap(df.corr(), annot = True, fmt= '.0%')


# In[15]:


X = df.drop(['diagnosis'], axis=1)
Y = df.diagnosis.values


# In[16]:


#Scaling values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


# In[17]:


print(X)


# In[18]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[19]:


#support vector classifier
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(X_train, y_train)
print("SVC accuracy : {:.2f}%".format(svm.score(X_test, y_test)*100))


# In[20]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
print("Naive Bayes accuracy : {:.2f}%".format(nb.score(X_test, y_test)*100))


# In[21]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 1000, random_state = 1)
rf.fit(X_train, y_train)
print("Random Forest accuracy : {:.2f}%".format(rf.score(X_test, y_test)*100))


# In[22]:


import xgboost
xg = xgboost.XGBClassifier()
xg.fit(X_train, y_train)
print("XG boost accuracy : {:.2f}%".format(xg.score(X_test, y_test)*100))


# In[23]:


#SVC accuracy : 98.25%
#Naive Bayes accuracy : 98.23%
#Random Forest accuracy : 95.61%
#XG boost accuracy : 98.25%


# In[ ]:




