#!/usr/bin/env python
# coding: utf-8

# # Parkison's Disease Detection

# ### importing dependencies 

# In[100]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as ac
from sklearn.preprocessing import StandardScaler
from sklearn import svm


# ### data collection and analysis

# 
# #### loading data from csv file to a Pandas DataFrame
# 
# 

# In[101]:


parkinsons_data = pd.read_csv('parkinsons.csv')


# 
# ### printing head (first 5 rows) of dataframe
# 
# 

# In[102]:


parkinsons_data.head()


# ### rows and columns

# In[103]:


parkinsons_data.shape 


# ### info check

# In[104]:


parkinsons_data.info()


# ### checking for missing values

# In[105]:


parkinsons_data.isnull().sum()


# ### stats on the given data

# In[106]:


parkinsons_data.describe()


# ### target variable distribution 

# In[107]:


parkinsons_data['status'].value_counts()

0  --> Healthy
1  --> Parkinson's Positive 
# ### data grouping based on target varible

# In[108]:


#parkinsons_data.groupby('status').mean()

Data Pre-ProcessingSeparating the Features & Target
# In[109]:


x = parkinsons_data.drop(columns=['name', 'status'], axis = 1)
y = parkinsons_data['status']


# In[110]:


print(x)


# In[111]:


print(y)


# ### splitting the data to training data & test data

# In[112]:


x_train, x_test, y_train, y_test = tts(x,y,test_size=.2,random_state = 2)


# In[113]:


print  (x.shape,x_train.shape, x_test.shape)


# ### data standardization

# In[114]:


scaler = StandardScaler()


# In[115]:


scaler.fit(x_train)


# In[116]:


x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)


# In[117]:


print(x_train)

Model Training

Support Vector Machine Model
# In[118]:


model = svm.SVC(kernel = 'linear')


# ### training SVM model with training data

# In[119]:


model.fit(x_train, y_train)


# ## Model Evaluation 
# 
# 

# ### Accuracy Score

# #### Accuracy score on training data

# In[120]:


x_train_prediction = model.predict(x_train)
training_data_accuracy = ac(y_train, x_train_prediction)


# In[121]:


print("Accuracy score of the training data: ",training_data_accuracy)


# #### Accuracy of the testing data

# In[122]:


x_test_prediction = model.predict(x_test)
testing_data_accuracy = ac(y_test, x_test_prediction)


# In[123]:


print("Accuracy score of the training data: ",training_data_accuracy)


# ### Building a Predective System

# In[124]:


input_data = (198.45800,219.29000,148.69100,0.00376,0.00002,0.00182,0.00215,0.00546,0.03527,0.29700,0.02055,0.02076,0.02530,0.06165,0.01728,18.70200,0.606273,0.661735,-5.585259,0.310746,2.465528,0.209863)


# ### changing input data to numpy array

# In[125]:


input_data_as_numpy_array = np.asarray(input_data)
                                                        


# #### reshape the numpy array

# In[126]:


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# ### standardize the data

# In[127]:


std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)

if prediction == [1]:
    print("This person has Parkinson's Disease")
else:
    print("NO")    



# In[ ]:





# In[ ]:





# In[ ]:




