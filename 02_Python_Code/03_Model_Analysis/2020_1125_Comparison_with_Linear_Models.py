#!/usr/bin/env python
# coding: utf-8

# # Predicting Concrete Compressive Strength - Comparison with Linear Models

# In this code notebook, we will analyze the statistics pertaining the various models presented in this project. In the Exploratory Data Analysis notebook, we explored the various relationships that each consituent of concrete has on the cured compressive strength. The materials that held the strongest relationships, regardless of curing time, were cement, cementitious ratio, superplasticizer ratio, and fly ash ratio. We will examine each of the linear ratios independent of age, as well as at the industry-standard 28 day cure time mark.

# ## Dataset Citation

# This dataset was retrieved from the UC Irvine Machine Learning Repository from the following URL: <https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength>. 
# 
# The dataset was donated to the UCI Repository by Prof. I-Cheng Yeh of Chung-Huah University, who retains copyright for the following published paper: I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998). Additional papers citing this dataset are listed at the reference link above.

# ## Import the Relevant Libraries

# In[1]:


# Data Manipulation
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Linear Regresssion Model
from sklearn.linear_model import LinearRegression

# Model Evaluation
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score


# ## Import & Check the Data

# In[2]:


df1 = pd.read_csv('2020_1124_Modeling_Data.csv')
df2 = pd.read_csv('2020_1123_Concrete_Data_Loaded_Transformed.csv')

original_data = df1.copy()
transformed_data = df2.copy()


# In[3]:


# The original data contains kg/m^3 values
original_data.head()


# In[4]:


# Original data
original_data.describe()


# In[5]:


# The transformed data contains ratios to total mass of the concrete mix
transformed_data.head()


# In[6]:


# Transformed data
transformed_data.describe()


# ## Cement Modeling - Including All Cure Times

# We understand that the ratio of cement to compressive strength is linear. We will model this relationship in Python and evaluate its performance compared to our ANN model.

# ### Visualization

# In[7]:


# We will visualize the linear relationship between quantity of cement and compressive strength
cement = original_data['Cement']
strength = original_data['Compressive_Strength']
plt.scatter(cement,strength)


# ### Train the Linear Model

# In[8]:


# Reshape the data so it complies with the linear model requirements
X = np.array(cement).reshape(1030,1)
y = np.array(strength).reshape(1030,1)


# In[9]:


# Perform a train-test split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the linear model
lm = LinearRegression()
lm.fit(X_train,y_train)


# ### Test the Linear Model

# In[10]:


y_pred = lm.predict(X_test)


# ### Linear Equation

# In[11]:


# print the intercept
print(lm.intercept_)


# In[12]:


coeff = pd.DataFrame(lm.coef_,columns=['Coefficient'])
coeff


# ### Model Evaluation

# In[13]:


# Plot the linear model preditions as a line superimposed on a scatter plot of the testing data
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,'r')


# In[14]:


# Evaluation Metrics
MAE_cement = mean_absolute_error(y_test, y_pred)
MSE_cement = mean_squared_error(y_test, y_pred)
RMSE_cement = np.sqrt(mean_squared_error(y_test, y_pred))

cement_stats = [MAE_cement,MSE_cement,RMSE_cement] # storing for model comparison at the end of this notebook

# Print the metrics
print(f"EVALUATION METRICS, LINEAR MODEL FOR CEMENT VS. COMPRESSIVE STRENGTH")
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_cement}\nMean Squared Error:\t\t\t{MSE_cement}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_cement}")
print('-----------------------------\n\n')


# ## Cement Modeling - 28 Day Cure Time

# We will model the cement vs compressive strength relationship for a constant cure time (28 days).

# ### Visualization

# In[15]:


# We will visualize the linear relationship between quantity of cement and compressive strength at 28 days
cement = original_data[original_data['Age']==28]['Cement']
strength = original_data[original_data['Age']==28]['Compressive_Strength']
plt.scatter(cement,strength)


# ### Train the Linear Model

# In[16]:


# Reshape the data so it complies with the linear model requirements
X = np.array(cement).reshape(425,1)
y = np.array(strength).reshape(425,1)


# In[17]:


# Perform a train-test split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the linear model
lm = LinearRegression()
lm.fit(X_train,y_train)


# ### Test the Linear Model

# In[18]:


y_pred = lm.predict(X_test)


# ### Linear Equation

# In[19]:


# print the intercept
print(lm.intercept_)


# In[20]:


coeff = pd.DataFrame(lm.coef_,columns=['Coefficient'])
coeff


# ### Model Evaluation

# In[21]:


# Plot the linear model preditions as a line superimposed on a scatter plot of the testing data
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,'r')


# In[22]:


# Evaluation Metrics
MAE_cement_28 = mean_absolute_error(y_test, y_pred)
MSE_cement_28 = mean_squared_error(y_test, y_pred)
RMSE_cement_28 = np.sqrt(mean_squared_error(y_test, y_pred))

cement_28_stats = [MAE_cement_28,MSE_cement_28,RMSE_cement_28] # storing for model comparison at the end of this notebook

# Print the metrics
print(f"EVALUATION METRICS, LINEAR MODEL FOR CEMENT VS. COMPRESSIVE STRENGTH")
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_cement_28}\nMean Squared Error:\t\t\t{MSE_cement_28}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_cement_28}")
print('-----------------------------\n\n')


# ## Cementitious Ratio Modeling - Including All Cure Times

# We know that the ratio of cementitious materials to the total mass is (cement + fly ash)/(total mass) to compressive strength is linear. We will model this relationship in Python and evaluate its performance.

# ### Visualization

# In[23]:


# We will visualize the linear relationship between quantity of cementitious materials and compressive strength
cementitious = transformed_data['Cementitious_Ratio']
strength = transformed_data['Compressive_Strength']
plt.scatter(cementitious,strength)


# ### Train the Linear Model

# In[24]:


# Reshape the data so it complies with the linear model requirements
X = np.array(cementitious).reshape(1030,1)
y = np.array(strength).reshape(1030,1)


# In[25]:


# Perform a train-test split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the linear model
lm = LinearRegression()
lm.fit(X_train,y_train)


# ### Test the Linear Model

# In[26]:


y_pred = lm.predict(X_test)


# ### Linear Equation

# In[27]:


# print the intercept
print(lm.intercept_)


# In[28]:


coeff = pd.DataFrame(lm.coef_,columns=['Coefficient'])
coeff


# ### Model Evaluation

# In[29]:


# Plot the linear model preditions as a line superimposed on a scatter plot of the testing data
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,'r')


# In[30]:


# Evaluation Metrics
MAE_cementitious = mean_absolute_error(y_test, y_pred)
MSE_cementitious = mean_squared_error(y_test, y_pred)
RMSE_cementitious = np.sqrt(mean_squared_error(y_test, y_pred))

cementitious_stats = [MAE_cementitious,MSE_cementitious,RMSE_cementitious] # storing for model comparison at the end of this notebook

# Print the metrics
print(f"EVALUATION METRICS, LINEAR MODEL FOR CEMENTITIOUS RATIO VS. COMPRESSIVE STRENGTH")
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_cementitious}\nMean Squared Error:\t\t\t{MSE_cementitious}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_cementitious}")
print('-----------------------------\n\n')


# ## Cementitious Ratio Modeling - 28 Day Cure Time

# ### Visualization

# In[31]:


# We will visualize the linear relationship between quantity of cementitious materials and compressive strength at 28 days
cementitious = transformed_data[original_data['Age']==28]['Cementitious_Ratio']
strength = transformed_data[original_data['Age']==28]['Compressive_Strength']
plt.scatter(cementitious,strength)


# ### Train the Linear Model

# In[32]:


# Reshape the data so it complies with the linear model requirements
X = np.array(cementitious).reshape(425,1)
y = np.array(strength).reshape(425,1)


# In[33]:


# Perform a train-test split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the linear model
lm = LinearRegression()
lm.fit(X_train,y_train)


# ### Test the Linear Model

# In[34]:


y_pred = lm.predict(X_test)


# ### Linear Equation

# In[35]:


# print the intercept
print(lm.intercept_)


# In[36]:


coeff = pd.DataFrame(lm.coef_,columns=['Coefficient'])
coeff


# ### Model Evaluation

# In[37]:


# Plot the linear model preditions as a line superimposed on a scatter plot of the testing data
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,'r')


# In[38]:


# Evaluation Metrics
MAE_cementitious_28 = mean_absolute_error(y_test, y_pred)
MSE_cementitious_28 = mean_squared_error(y_test, y_pred)
RMSE_cementitious_28 = np.sqrt(mean_squared_error(y_test, y_pred))

cementitious_28_stats = [MAE_cementitious_28,MSE_cementitious_28,RMSE_cementitious_28] # storing for model comparison at the end of this notebook

# Print the metrics
print(f"EVALUATION METRICS, LINEAR MODEL FOR CEMENTITIOUS RATIO VS. COMPRESSIVE STRENGTH AT 28 DAYS")
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_cementitious_28}\nMean Squared Error:\t\t\t{MSE_cementitious_28}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_cementitious_28}")
print('-----------------------------\n\n')


# ## Fly Ash Ratio Modeling - Including All Cure Times

# The fly ash ratio is interpreted as the percentage of fly ash within the cementitious materials mix, that is, Fly_Ash_Ratio = (fly ash + cement)/(total mass).

# ### Visualization

# In[39]:


# We will visualize the linear relationship between fly ash ratio and compressive strength
fly = transformed_data['Fly_Ash_Ratio']
strength = transformed_data['Compressive_Strength']
plt.scatter(fly,strength)


# ### Data Preprocessing

# We see from the graph above that there are many instances where there is no fly ash in the mix design. Let us use only nonzero entries for our analysis.

# In[40]:


fly = transformed_data[transformed_data['Fly_Ash_Ratio']!=0]['Fly_Ash_Ratio']
strength = transformed_data[transformed_data['Fly_Ash_Ratio']!=0]['Compressive_Strength']
plt.scatter(fly,strength)


# ### Train the Linear Model

# In[41]:


# Reshape the data so it complies with the linear model requirements
X = np.array(fly).reshape(464,1)
y = np.array(strength).reshape(464,1)


# In[42]:


# Perform a train-test split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the linear model
lm = LinearRegression()
lm.fit(X_train,y_train)


# ### Test the Linear Model

# In[43]:


y_pred = lm.predict(X_test)


# ### Linear Equation

# In[44]:


# print the intercept
print(lm.intercept_)


# In[45]:


coeff = pd.DataFrame(lm.coef_,columns=['Coefficient'])
coeff


# ### Model Evaluation

# In[46]:


# Plot the linear model preditions as a line superimposed on a scatter plot of the testing data
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,'r')


# In[47]:


# Evaluation Metrics
MAE_fly = mean_absolute_error(y_test, y_pred)
MSE_fly = mean_squared_error(y_test, y_pred)
RMSE_fly = np.sqrt(mean_squared_error(y_test, y_pred))

fly_stats = [MAE_fly,MSE_fly,RMSE_fly] # storing for model comparison at the end of this notebook

# Print the metrics
print(f"EVALUATION METRICS, LINEAR MODEL FOR FLY ASH RATIO VS. COMPRESSIVE STRENGTH")
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_fly}\nMean Squared Error:\t\t\t{MSE_fly}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_fly}")
print('-----------------------------\n\n')


# ## Fly Ash Ratio Modeling - 28 Day Cure Time

# The fly ash ratio is interpreted as the percentage of fly ash within the cementitious materials mix, that is, Fly_Ash_Ratio = (fly ash + cement)/(total mass).

# In[48]:


fly = transformed_data[((transformed_data['Fly_Ash_Ratio']!=0)&(transformed_data['Age']==28))]['Fly_Ash_Ratio']
strength = transformed_data[((transformed_data['Fly_Ash_Ratio']!=0)&(transformed_data['Age']==28))]['Compressive_Strength']
plt.scatter(fly,strength)


# ### Train the Linear Model

# In[49]:


# Reshape the data so it complies with the linear model requirements
X = np.array(fly).reshape(217,1)
y = np.array(strength).reshape(217,1)


# In[50]:


# Perform a train-test split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the linear model
lm = LinearRegression()
lm.fit(X_train,y_train)


# ### Test the Linear Model

# In[51]:


y_pred = lm.predict(X_test)


# ### Linear Equation

# In[52]:


# print the intercept
print(lm.intercept_)


# In[53]:


coeff = pd.DataFrame(lm.coef_,columns=['Coefficient'])
coeff


# ### Model Evaluation

# In[54]:


# Plot the linear model preditions as a line superimposed on a scatter plot of the testing data
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,'r')


# In[55]:


# Evaluation Metrics
MAE_fly_28 = mean_absolute_error(y_test, y_pred)
MSE_fly_28 = mean_squared_error(y_test, y_pred)
RMSE_fly_28 = np.sqrt(mean_squared_error(y_test, y_pred))

fly_28_stats = [MAE_fly_28,MSE_fly_28,RMSE_fly_28] # storing for model comparison at the end of this notebook

# Print the metrics
print(f"EVALUATION METRICS, LINEAR MODEL FOR FLY ASH RATIO VS. COMPRESSIVE STRENGTH AT 28 DAYS")
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_fly_28}\nMean Squared Error:\t\t\t{MSE_fly_28}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_fly_28}")
print('-----------------------------\n\n')


# ## Superplasticizer Ratio Modeling - Including All Cure Times

# The superplasticizer ratio is the ratio of superplasticizer contained within the total mix design, by weight.

# ### Visualization

# In[56]:


# We will visualize the linear relationship between superplasticizer ratio and compressive strength
superplasticizer = transformed_data['Superplasticizer_Ratio']
strength = transformed_data['Compressive_Strength']
plt.scatter(superplasticizer,strength)


# ### Data Preprocessing

# Once agaain, we see from the graph above that there are many instances where there is no superplasticizer in the mix design. Let us use only nonzero entries for our analysis.

# In[57]:


superplasticizer = transformed_data[transformed_data['Superplasticizer_Ratio']!=0]['Superplasticizer_Ratio']
strength = transformed_data[transformed_data['Superplasticizer_Ratio']!=0]['Compressive_Strength']
plt.scatter(superplasticizer,strength)


# This is better, but we see a large spread in the data. Let's remove any outliers first, before training our model.

# In[58]:


superplasticizer.describe()


# In[59]:


mean = 0.004146
three_sigma = 3*0.001875
upper = mean + three_sigma
lower = mean - three_sigma

print(f"The lower bound is:\t{lower}\nThe upper bound is:\t{upper}")


# Since there are no negative ratios, we only need to remove data points where the superplasticizer ratio is greater than 0.009771.

# In[60]:


superplasticizer = transformed_data[transformed_data['Superplasticizer_Ratio']!=0][transformed_data['Superplasticizer_Ratio'] < upper]['Superplasticizer_Ratio']
strength = transformed_data[transformed_data['Superplasticizer_Ratio']!=0][transformed_data['Superplasticizer_Ratio'] < upper]['Compressive_Strength']
plt.scatter(superplasticizer,strength)


# ### Train the Linear Model

# In[61]:


# We will train and test our model only on the data above, that does not contain outliers
# Reshape the data so it complies with the linear model requirements
X = np.array(superplasticizer).reshape(641,1)
y = np.array(strength).reshape(641,1)


# In[62]:


# Perform a train-test split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the linear model
lm = LinearRegression()
lm.fit(X_train,y_train)


# ### Test the Linear Model

# In[63]:


y_pred = lm.predict(X_test)


# ### Linear Equation

# In[64]:


# print the intercept
print(lm.intercept_)


# In[65]:


coeff = pd.DataFrame(lm.coef_,columns=['Coefficient'])
coeff


# ### Model Evaluation

# In[66]:


# Plot the linear model preditions as a line superimposed on a scatter plot of the testing data
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,'r')


# In[67]:


# Evaluation Metrics
MAE_super = mean_absolute_error(y_test, y_pred)
MSE_super = mean_squared_error(y_test, y_pred)
RMSE_super = np.sqrt(mean_squared_error(y_test, y_pred))

super_stats = [MAE_super,MSE_super,RMSE_super] # storing for model comparison at the end of this notebook

# Print the metrics
print(f"EVALUATION METRICS, LINEAR MODEL FOR SUPERPLASTICIZER RATIO VS. COMPRESSIVE STRENGTH")
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_super}\nMean Squared Error:\t\t\t{MSE_super}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_super}")
print('-----------------------------\n\n')


# ## Superplasticizer Ratio Modeling - 28 Day Cure Time

# The superplasticizer ratio is the ratio of superplasticizer contained within the total mix design, by weight.

# ### Visualization

# In[68]:


superplasticizer = transformed_data[((transformed_data['Superplasticizer_Ratio']!=0)&(transformed_data['Age']==28))]['Superplasticizer_Ratio']
strength = transformed_data[((transformed_data['Superplasticizer_Ratio']!=0)&(transformed_data['Age']==28))]['Compressive_Strength']
plt.scatter(superplasticizer,strength)


# This is better, but we see a large spread in the data. Let's remove any outliers first, before training our model.

# In[69]:


superplasticizer.describe()


# In[70]:


mean = 0.004146
three_sigma = 3*0.001875
upper = mean + three_sigma
lower = mean - three_sigma

print(f"The lower bound is:\t{lower}\nThe upper bound is:\t{upper}")


# Since there are no negative ratios, we only need to remove data points where the superplasticizer ratio is greater than 0.009771.

# In[71]:


superplasticizer = transformed_data[((transformed_data['Superplasticizer_Ratio']!=0)&(transformed_data['Age']==28)&(transformed_data['Superplasticizer_Ratio']<upper))]['Superplasticizer_Ratio']
strength = transformed_data[((transformed_data['Superplasticizer_Ratio']!=0)&(transformed_data['Age']==28)&(transformed_data['Superplasticizer_Ratio']<upper))]['Compressive_Strength']
plt.scatter(superplasticizer,strength)


# ### Train the Linear Model

# In[72]:


# We will train and test our model only on the data above, that does not contain outliers
# Reshape the data so it complies with the linear model requirements
X = np.array(superplasticizer).reshape(315,1)
y = np.array(strength).reshape(315,1)


# In[73]:


# Perform a train-test split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the linear model
lm = LinearRegression()
lm.fit(X_train,y_train)


# ### Test the Linear Model

# In[74]:


y_pred = lm.predict(X_test)


# ### Linear Equation

# In[75]:


# print the intercept
print(lm.intercept_)


# In[76]:


coeff = pd.DataFrame(lm.coef_,columns=['Coefficient'])
coeff


# ### Model Evaluation

# In[77]:


# Plot the linear model preditions as a line superimposed on a scatter plot of the testing data
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,'r')


# In[78]:


# Evaluation Metrics
MAE_super_28 = mean_absolute_error(y_test, y_pred)
MSE_super_28 = mean_squared_error(y_test, y_pred)
RMSE_super_28 = np.sqrt(mean_squared_error(y_test, y_pred))

super_stats_28 = [MAE_super_28,MSE_super_28,RMSE_super_28] # storing for model comparison at the end of this notebook

# Print the metrics
print(f"EVALUATION METRICS, LINEAR MODEL FOR SUPERPLASTICIZER RATIO VS. COMPRESSIVE STRENGTH AT 28 DAYS")
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_super_28}\nMean Squared Error:\t\t\t{MSE_super_28}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_super_28}")
print('-----------------------------\n\n')


# ## Model Comparisons Analysis

# Neither superplasticizer linear model appeared to represent the data well from a visual perspective. The cement, cementitious ratio, and fly ash ratio linear models, however, did. We can display all of the evaluation metrics below and compare them to the artificial neural network's (ANN) performance.

# In[103]:


ANN_metrics = [5.083552,6.466492**2,6.466492]

metrics = [cement_stats, cementitious_stats, fly_stats, super_stats, ANN_metrics]
metrics_28 = [cement_28_stats, cementitious_28_stats, fly_28_stats, super_stats_28, ANN_metrics]

metrics_df = pd.DataFrame(data=metrics, index=['Cement (Ignoring Cure Time)','Cementitious_Ratio (Ignoring Cure Time)','Fly_Ash_Ratio (Ignoring Cure Time)','Superplasticizer_Ratio (Ignoring Cure Time)','ANN (Function of Time)'], columns=['MAE','MSE','RMSE'])
metrics_28_df = pd.DataFrame(data=metrics_28, index=['Cement (Cure Time = 28 Days)','Cementitious_Ratio (Cure Time = 28 Days)','Fly_Ash_Ratio (Cure Time = 28 Days)','Superplasticizer_Ratio (Cure Time = 28 Days)','ANN (Function of Time)'], columns=['MAE','MSE','RMSE'])


# In[104]:


metrics_df


# In[105]:


metrics_28_df


# ## Conclusions & Recommendations

# By comparing the evaluation metrics for all models, we conclude that the ANN model performed significantly better than all of the linear models. It outperformed the best linear model's RMSE (for Fly_Ash_Ratio at 28 Days) by over 30%! An important note is that the linear models were not scaled, and the ANN model was. We kept the linear models biased in order to maintain coefficient interpretabililty, whereas that was not relevant to the ANN model.
# 
# What is surprising is that the ANN model still outperformed the linear models, even when controlling for cure time at 28 days. Perhaps the most startling insight is that the fly ash ratio was even more accurate at predicting concrete compressive strength than the cement quantity, to the point that it had the lowest errors of all of the linear models. We therefore recommend that engineers give very conservative fly ash ratio specifications when allowing substitutions for Portland cement.
