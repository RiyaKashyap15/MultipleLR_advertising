#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np #for array constructions and transformations
import pandas as  pd # for data manipulation
import matplotlib.pyplot as plt# for creating static, animated, and interactive visualizations
import seaborn as sns #for data visualization
from sklearn.linear_model import LinearRegression #for classification, predictive analytics, and very many other machine learning tasks
from sklearn.metrics import mean_squared_error, r2_score
import os


# In[3]:


os.getcwd() # to get current working directory


# In[5]:


df= pd.read_csv("Downloads/advertising.csv")


# In[6]:


df.head()


# In[7]:


#Setting the value for X and Y
x = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


# In[8]:


#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[9]:


#Fitting the Multiple Linear Regression model
mlr = LinearRegression()  
mlr.fit(x_train, y_train)


# In[10]:


#Intercept and Coefficient
print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(x, mlr.coef_))


# In[11]:


#Prediction of test set
y_pred_mlr= mlr.predict(x_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))


# In[14]:


#Actual value and the predicted value
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff.head()


# In[15]:


#Model Evaluation
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# In[ ]:




