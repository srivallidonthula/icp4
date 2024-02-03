#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd


# In[23]:


# 1(a) Import the given “Data.csv”
dst_Data = pd.read_csv(r"C:\Users\dsriv\OneDrive\Desktop\data.csv")
dst_Data.info()


# In[24]:


#(c) Show the basic statistical description about the data.
dst_Data.head()


# In[25]:


#(d)Check if the data has null values.
dst_Data.isnull().any()


# In[27]:


dst_Data.fillna(dst_Data.mean(), inplace=True)
dst_Data.isnull().any()


# In[26]:


#d(i)Replace the null values with the mean
column_means = dst_Data.mean()
print(column_means)
dst_Data = dst_Data. fillna(column_means)
print(dst_Data.head(20))


# In[8]:


#(e)Select at least two columns and aggregate the data using: min, max, count, mean.
res = dst_Data.agg({'Calories': ['mean', 'min','max', 'count'],'Pulse': ['mean', 'min', 'max', 'count']})
print(res)


# In[9]:


#(f)Filter the dataframe to select the rows with calories values between 500 and 1000
filter_dst_Data1=dst_Data[(dst_Data['Calories'] > 500) & (dst_Data['Calories'] < 1000)]
print(filter_dst_Data1)
#(g)Filter the dataframe to select the rows with calories values > 500 and pulse < 100.
filter_dst_Data2=dst_Data[(dst_Data['Calories'] > 500) & (dst_Data['Pulse'] < 100)]
print(filter_dst_Data2)


# In[10]:


#(h)Create a new “df_modified” dataframe that contains all the columns from dst_data except for
#“Maxpulse”.
df_modified = dst_Data.loc[:, dst_Data.columns != 'Maxpulse']
print(df_modified)


# In[11]:


#(i). Delete the “Maxpulse” column from the main dst_data dataframe
dst_Data.drop('Maxpulse', inplace=True, axis=1)
print(dst_Data.dtypes)


# In[12]:


#(j). Convert the datatype of Calories column to int datatype
dst_Data["Calories"] = dst_Data["Calories"].astype(int)
print(dst_Data.dtypes)


# In[30]:


#(k)Using pandas create a scatter plot for the two columns (Duration and Calories).
as1 = dst_Data.plot.scatter(x='Duration',y='Calories')
print(as1)


# In[16]:


# 2(a) Import the given “Salary_Data.csv”
dst_Sal = pd.read_csv(r"C:\Users\dsriv\OneDrive\Desktop\Salary_Data.csv")
dst_Sal.info()
dst_Sal.head()


# In[17]:


A = dst_Sal.iloc[:, :-1].values   #excluding last column i.e., years of experience column
B = dst_Sal.iloc[:, 1].values     #only salary column


# In[18]:


# (b) Split the data in train_test partitions, such that 1/3 of the data is reserved as test subset.
from sklearn.model_selection import train_test_split 
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=1/3, random_state=0)


# In[19]:


# (c) Train and predict the model.
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(A_train, B_train)
B_Pred = reg.predict(A_test)
B_Pred


# In[28]:


# (d) Calculate the mean_squared error
S_error = (B_Pred - B_test) ** 2
Sum_Serror = np.sum(S_error)
mean_squared_error = Sum_Serror / B_test.size
mean_squared_error


# In[29]:


# (e) Visualize both train and test data using scatter plot.
import matplotlib.pyplot as plt
# Training Data set
plt.scatter(A_train, B_train)
plt.plot(A_train, reg.predict(A_train), color='red')
plt.title('Training Set')
plt.show()

# Testing Data set
plt.scatter(A_test, B_test)
plt.plot(A_test, reg.predict(A_test), color='red')
plt.title('Testing Set')
plt.show()


# In[ ]:




