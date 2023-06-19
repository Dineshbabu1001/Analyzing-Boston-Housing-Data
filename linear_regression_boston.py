#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go


# In[3]:


# Load the dataset
url = "Boston.csv"
Boston_Data=pd.read_csv(url)


# In[4]:


# Display the first few rows of the dataset
print(Boston_Data.head())


# In[5]:


# Perform exploratory data analysis (EDA)
print(Boston_Data.describe())


# In[6]:


# Check for missing values
print(Boston_Data.isnull().sum())


# In[7]:


# Analyze the distribution of the target variable (charges)
sns.displot(Boston_Data['medv'])
plt.title('Distribution of medv')
plt.show()


# In[8]:


# Analyze the relationship between variables
sns.pairplot(Boston_Data)
plt.show()


# In[9]:


# Analyze the correlation between variables
correlation = Boston_Data.corr()
sns.heatmap(correlation, annot=False)
plt.title('Correlation Matrix')
plt.show()


# In[10]:


# Convert categorical variables into numerical form (dummy encoding)
Boston_Data = pd.get_dummies(Boston_Data, drop_first=True)


# In[11]:


# Split the data into training and testing sets
X = Boston_Data.drop(['Unnamed: 0','medv'], axis=1)
y = Boston_Data['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[12]:


# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[13]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[14]:


# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[15]:


# Split the data into independent variables (X) and the target variable (y)
X = Boston_Data.drop(['Unnamed: 0','medv'], axis=1)
y = Boston_Data['medv']


# In[16]:


# Add a constant column to X
X = sm.add_constant(X)


# In[17]:


# Fit the OLS model
model = sm.OLS(y, X)
results = model.fit()


# In[18]:


# Print the summary of the model
print(results.summary())


# In[19]:


# Fit a best-fit line using numpy's polyfit function
fit = np.polyfit(Boston_Data['age'], Boston_Data['medv'], deg=1)
line = np.poly1d(fit)


# In[20]:


# Create the scatter plot
scatter_plot = px.scatter(Boston_Data, x='age', y='medv', trendline='ols')


# In[21]:


# Create the best-fit line trace
best_fit_line = go.Scatter(x=Boston_Data['age'], y=line(Boston_Data['age']), mode='lines', name='Best Fit Line')


# In[22]:


# Add the best-fit line to the scatter plot
scatter_plot.add_trace(best_fit_line)
# Show the scatter plot
scatter_plot.show()


# In[ ]:




