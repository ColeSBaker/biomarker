#!/usr/bin/env python
# coding: utf-8

# In[4]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os

import pandas as pd
from data_collection import *
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


NUM_TEST = 20
SEED = 1


# In[2]:


np.random.seed(SEED)


# In[3]:


excel = parse_master_file()


# In[1]:


L = get_filename_list(excel['Associated data'])


# In[6]:


x1 = create_x1_matrix(L)


# In[7]:


x1.shape


# In[8]:


x4 = create_x4_matrix(L)
# print(x4)
# In[9]:


# x4.shape


# In[10]:


x5 = create_x5_matrix(L)
# print(x5)

# In[11]:


x5.shape


# In[12]:


x6 = create_x6_matrix(L)
# print(x6)

# In[13]:


x6.shape


# In[14]:


x7 = create_x7_matrix(L)
# print(x7)

# In[15]:


x7.shape


# In[16]:


y = excel['Output: logK'].values
x10_x17 = excel.iloc[:, 3:-2]


# In[17]:


master = prepare_master(x10_x17)


# In[18]:


test_idxs = np.random.randint(0,len(y),NUM_TEST)
train_idxs = np.ones(y.shape,dtype=bool)
train_idxs[test_idxs] = False
y_train = y[train_idxs]
y_test = y[test_idxs]


# # In[19]:
# print(str(x1[train_idxs],"X!111111111111"))

x1_approx_train = linear_regression_approx(x1[train_idxs], y_train)
x4_approx_train = linear_regression_approx(x4[train_idxs], y_train)
x5_approx_train = linear_regression_approx(x5[train_idxs], y_train)
x6_approx_train = linear_regression_approx(x6[train_idxs], y_train)
x7_approx_train = linear_regression_approx(x7[train_idxs], y_train)


# In[20]:


master_train = master[train_idxs]
master_test = master[test_idxs]


# In[ ]:





# In[21]:


regr = linear_model.LinearRegression()
all_xs_train = np.column_stack((x1_approx_train, x4_approx_train, x5_approx_train, x6_approx_train, x7_approx_train, master_train))
regr.fit(all_xs_train, y_train)


# In[22]:


x1_approx_test = linear_regression_approx(x1[test_idxs], y_test)
x4_approx_test = linear_regression_approx(x4[test_idxs], y_test)
x5_approx_test = linear_regression_approx(x5[test_idxs], y_test)
x6_approx_test = linear_regression_approx(x6[test_idxs], y_test)
x7_approx_test = linear_regression_approx(x7[test_idxs], y_test)


# In[23]:


all_xs_test = np.column_stack((x1_approx_test, x4_approx_test, x5_approx_test, x6_approx_test, x7_approx_test, master_test))


# In[24]:


predictions = regr.predict(all_xs_test)


# In[25]:


predictions - y_test


# In[26]:


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, predictions))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, predictions))

# # Plot outputs
# plt.scatter(diabetes_X_test, y_test,  color='black')
# plt.plot(diabetes_X_test, predictions, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()

