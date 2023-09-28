#!/usr/bin/env python
# coding: utf-8

# In[181]:


#import libraray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[182]:


#loading the data set
dataset=pd.read_csv('Salary_Data.csv')
dataset.head()


# In[183]:


# Assuming that 'dataset' contains your data
x = dataset.iloc[:, :-1].values  # Features (all columns except the last one)
y = dataset.iloc[:, -1].values   # Target variable (last column)


# In[184]:


x


# In[185]:


y


# In[186]:


print("x shape:", x.shape)
print("y shape:", y.shape)


# In[187]:


plt.scatter(x,y)
plt.xlabel("inDependend Varible")
plt.ylabel("Depended varible")
plt.show()


# In[188]:


#spliting the data set into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:






# In[189]:


#training the model on model set
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[190]:


#predecting test result
y_pred_test=reg.predict(x_test)
y_pred_train=reg.predict(x_train)


# In[191]:


plt.scatter(x_train,y_train,color="Red")
plt.plot(x_train,y_pred_train,color="blue")
plt.title("salary vs Exp training set")
plt.xlabel("year of exp")
plt.ylabel('Salary')
plt.show()


# In[192]:


#valuzing the test result
plt.scatter(x_test,y_test,color="Red")
plt.plot(x_test,y_pred_test,color="blue")
plt.title("salary vs Exp training set")
plt.xlabel("year of exp")
plt.ylabel('Salary')
plt.show()


# In[195]:


input_features = np.array([[10.8]])
reg.predict(input_features)

#predecting test result
new=reg.predict(input_features)
new


# In[194]:


import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred_test), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred_test), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred_test), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred_test), 2))
print("R2 score =", round(sm.r2_score(y_test, y_pred_test)))


# In[ ]:





# In[ ]:





# In[ ]:




