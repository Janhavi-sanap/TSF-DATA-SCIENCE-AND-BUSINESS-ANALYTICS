#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation -GRIP (May 2021)
# 
# Task 1:Prediction using Supervised Machine Learning
# 
# By: Janhavi Sanap

# In[1]:


#import the libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


#load the dataset 

url = ("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
data = pd.read_csv(url)
print("The data is successfully loaded.")
data


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


# For statistical function.

data.describe()


# In[6]:


data.info()


# # Vizualizing the data

# In[7]:


sb.pairplot(data)
plt.show()


# In[8]:


X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values


# # Preparing Data and splitting into train and test datasets.

# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0,test_size=0.2)


# In[10]:


# We have Splitted Our Data Using 80:20 Rule

print("X train.shape =", X_train.shape)
print("Y train.shape =", Y_train.shape)
print("X test.shape  =", X_test.shape)
print("Y test.shape  =", Y_test.shape)

TRAINING A MODEL
# In[11]:


from sklearn.linear_model import LinearRegression
linr=LinearRegression()
linr.fit(X_train,Y_train)
print("Training our algorithm is finished")


# In[12]:


# β0 is Intercept & Slope of the line is β1

print("B0 =",linr.intercept_,"\nB1 =",linr.coef_)


# In[13]:


# Plotting the REGRESSION LINE

Y0 = linr.intercept_ + linr.coef_*X_train


# In[14]:


# Plotting on training data

plt.scatter(X_train,Y_train,color='blue',marker='+')
plt.plot(X_train,Y0,color='orange')
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Train set)",fontsize=10)
plt.show()


# In[15]:


# To predict the scores of testing data 

Y_pred=linr.predict(X_test)
print(Y_pred)


# In[16]:


Y_test


# In[17]:


# Plotting the Regression line on testing data

plt.plot(X_test,Y_pred,color='red')
plt.scatter(X_test,Y_test,color='green',marker='+')
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Test set)",fontsize=10)
plt.show()


# In[18]:


Y_test1 = list(Y_test)
prediction=list(Y_pred)
dt_compare = pd.DataFrame({ 'Actual':Y_test1,'Result':prediction})
dt_compare


# In[19]:


from sklearn import metrics
metrics.r2_score(Y_test,Y_pred)


# # Above 94% indicates that above fitted Model is a GOOD MODEL.

# Now, Predicting the Error:

# In[23]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[24]:


# To find the mean squared error, root mean squared error

mse = metrics.mean_squared_error(Y_test,Y_pred)
rmse = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
print("Mean Squared Error      = ", mse)
print("Root Mean Squared Error = ",rmse)


# # Predicting the Scores for 9.25 hrs:

# In[25]:


Prediction_score = linr.predict([[9.25]])
print("Predicted score for a student studying 9.25 hours :",Prediction_score)


# # CONCLUSION:

# # From the above result we can say that if a student  studied for 9.25 hrs then the student will scored 93.69%

# # Successfully completed the task.

# In[ ]:




