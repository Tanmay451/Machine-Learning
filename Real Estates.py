#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split


# In[2]:


housing = pd.read_csv("https://raw.githubusercontent.com/Tanmay451/Real-Estate/master/housing_data.csv")
housing.head()


# In[3]:


housing.info()


# In[4]:


housing['CHAS'].value_counts()


# In[5]:


housing.describe()


# In[6]:


housing['ZN'].describe()


# In[7]:


housing.hist(bins=50, figsize=(30, 20))


# ### Train-Test splitting
Only for learning purpos, we will use sklearn for real world problem.....
So comment it out


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set, test_set = split_train_test(housing,0.2)

print(len(train_set))
print(len(test_set))
# In[19]:


train_set , test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

it is for random shuffl but we want to shuffl CHAS equall in both set so we use bellow command
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# ### Looking for Correlations

# In[27]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
# to see if MEDV decrease then all other behaviar like if MEDV decrease then 69% of RM will decrease


# In[29]:


# to plot graphs
from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[31]:


# for only two attribut is wanna see more then one attribut then follow above
housing.plot(kind="scatter", x="RM", y= "MEDV", alpha=1)


# ### Trying out attribute combinations

# In[35]:


# adding some attributes 
housing["TAXRM"] = housing["TAX"]/housing["RM"]
housing.plot(kind="scatter", x="RM", y= "TAXRM", alpha=1)

To take care missing attributs

we can do 3 things

1. get rid of the missing data points

2. make it mean median or mode 
find median as:
    median = housing["RM"].median()
and fill the missing attribut with it
    housing["RM"].fillna(median)

you can find more in SimpleImputer form sklear.impute
# ### Scikit-learn Design

# It has Primarily, three type of objects
# 1. Estimator
#     It estimates some parameter based on a dataset
# 
# 2. Transformers
#     Method take input and give output based on the learning from fit().
# 
# 3. Predictors
#     You know

# In[ ]:




