#!/usr/bin/env python
# coding: utf-8

# # Predicting Concrete Compressive Strength - Artificial Neural Network (ANN) Modeling in TensorFlow 2.0

# In this code notebook, we will import the data, scale it, perform a train-test split, and run it through various artificial neural network (ANN) configurations in TensorFlow 2.0 using Keras.

# ## Dataset Citation

# This dataset was retrieved from the UC Irvine Machine Learning Repository from the following URL: <https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength>. 
# 
# The dataset was donated to the UCI Repository by Prof. I-Cheng Yeh of Chung-Huah University, who retains copyright for the following published paper: I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998). Additional papers citing this dataset are listed at the reference link above.

# ## Import the Relevant Libraries

# In[2]:


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

# ANN Modeling in TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

# Model Evaluation
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score


# ## Data Preprocessing

# ### Import & Check the Data

# In[3]:


df = pd.read_csv('2020_1124_Modeling_Data.csv')
concrete_data = df.copy()


# In[4]:


concrete_data.head()


# ### Train Test Split

# In[12]:


X = concrete_data.drop('Compressive_Strength',axis=1)
y = concrete_data['Compressive_Strength']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# ### Scale the Data

# In[16]:


scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Model 1 - ANN with 3 Hidden Layers

# ### Construct the Artificial Neural Network

# In[19]:


# Determine number of starting nodes by finding the shape of X_train
X_train.shape


# In[20]:


model = Sequential()

# We will start with 2 hidden layers. 
model.add(Dense(8,activation='relu')) # All layers utilize rectified linear units (relu)
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse') # Use the adam optimization algorithm


# ### Train the Model on the Test Data

# In[21]:


model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400)


# ### Visualize the Loss Function

# In[22]:


losses = pd.DataFrame(model.history.history)
losses.plot()


# Since the validation loss stays below the actual loss and continues to delcine with it, we observe that overfitting is minimal.

# ### Test the Model

# In[23]:


predictions = model.predict(X_test)


# ### Model Evaluation

# In[27]:


# Model Evaluation Metrics
MAE = mean_absolute_error(y_test,predictions)
RMSE = np.sqrt(mean_squared_error(y_test,predictions))
EVS = explained_variance_score(y_test,predictions)

print('EVALUATION METRICS')
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE}\nRoot Mean Squared Error (RMSE):\t\t{RMSE}\nExplained Variance Score:\t\t{EVS}")


# In[28]:


# Plot Model Predictions (Scatter)
plt.scatter(y_test,predictions)

# Plot Perfect predictions (Line)
plt.plot(y_test,y_test,'r')


# There is clearly a wide spread of predicted values away from the perfect values. Let us experiment with adding more hidden nodes in the next section to try to increase the performance of our model.

# ## Model 2 - ANN with 10 Hidden Layers

# ### Construct the Artificial Neural Network

# In[34]:


model_2 = Sequential()

# Experiment with 10 hidden layers
model_2.add(Dense(8,activation='relu')) # All layers utilize rectified linear units (relu)
model_2.add(Dense(8,activation='relu'))
model_2.add(Dense(8,activation='relu'))
model_2.add(Dense(8,activation='relu'))
model_2.add(Dense(8,activation='relu'))
model_2.add(Dense(8,activation='relu'))
model_2.add(Dense(8,activation='relu'))
model_2.add(Dense(8,activation='relu'))
model_2.add(Dense(8,activation='relu'))
model_2.add(Dense(4,activation='relu')) # Experiment with number of nodes
model_2.add(Dense(2,activation='relu'))
model_2.add(Dense(1))

model_2.compile(optimizer='adam',loss='mse') # Use the adam optimization algorithm


# ### Train the Model on the Test Data

# In[35]:


model_2.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400)


# ### Visualize the Loss Function

# In[36]:


losses = pd.DataFrame(model_2.history.history)
losses.plot()


# Again, we do not observe overfitting on the training data.

# ### Test the Model

# In[40]:


predictions_2 = model_2.predict(X_test)


# ### Model Evaluation

# In[41]:


# Model Evaluation Metrics
MAE_2 = mean_absolute_error(y_test,predictions_2)
RMSE_2 = np.sqrt(mean_squared_error(y_test,predictions_2))
EVS_2 = explained_variance_score(y_test,predictions_2)

print('EVALUATION METRICS')
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_2}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_2}\nExplained Variance Score:\t\t{EVS_2}")


# In[42]:


# Plot Model Predictions (Scatter)
plt.scatter(y_test,predictions_2)

# Plot Perfect predictions (Line)
plt.plot(y_test,y_test,'r')


# The variance of our predicted values has been decreased, and our explained variance score has increased significantly. Let us continue with an even deeper neural network below to see if it will increase performance.

# ## Model 3 - ANN with 20 Hidden Layers

# ### Construct the Artificial Neural Network

# In[43]:


model_3 = Sequential()

# Experiment with 20 hidden layers
model_3.add(Dense(8,activation='relu')) # All layers utilize rectified linear units (relu)
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(4,activation='relu')) # Experiment with number of nodes
model_3.add(Dense(2,activation='relu'))
model_3.add(Dense(1))

model_3.compile(optimizer='adam',loss='mse') # Use the adam optimization algorithm


# ### Train the Model on the Test Data

# In[44]:


model_3.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400)


# ### Visualize the Loss Function

# In[45]:


losses = pd.DataFrame(model_3.history.history)
losses.plot()


# ### Test the Model

# In[46]:


predictions_3 = model_3.predict(X_test)


# ### Model Evaluation

# In[50]:


# Model Evaluation Metrics
MAE_3 = mean_absolute_error(y_test,predictions_3)
RMSE_3 = np.sqrt(mean_squared_error(y_test,predictions_3))
EVS_3 = explained_variance_score(y_test,predictions_3)

print('EVALUATION METRICS')
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_3}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_3}\nExplained Variance Score:\t\t{EVS_3}")


# In[51]:


# Plot Model Predictions (Scatter)
plt.scatter(y_test,predictions_3)

# Plot Perfect predictions (Line)
plt.plot(y_test,y_test,'r')


# Interesting - we would expect from the loss function that the data was not overfitted. But Our model evaluation metrics are worse with the deeper neural network. It appears that Model 3 has overfitted to the training data.

# ## Model 4 - ANN with 15 Hidden Layers

# ### Construct the Artificial Neural Network

# In[52]:


model_4 = Sequential()

# Experiment with 15 hidden layers
model_4.add(Dense(8,activation='relu')) # All layers utilize rectified linear units (relu)
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(8,activation='relu'))
model_4.add(Dense(4,activation='relu')) # Experiment with number of nodes
model_4.add(Dense(2,activation='relu'))
model_4.add(Dense(1))

model_4.compile(optimizer='adam',loss='mse') # Use the adam optimization algorithm


# ### Train the Model on the Test Data

# In[53]:


model_4.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400)


# ### Visualize the Loss Function

# In[54]:


losses = pd.DataFrame(model_4.history.history)
losses.plot()


# ### Test the Model

# In[55]:


predictions_4 = model_4.predict(X_test)


# ### Model Evaluation

# In[57]:


# Model Evaluation Metrics
MAE_4 = mean_absolute_error(y_test,predictions_4)
RMSE_4 = np.sqrt(mean_squared_error(y_test,predictions_4))
EVS_4 = explained_variance_score(y_test,predictions_4)

print('EVALUATION METRICS')
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_4}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_4}\nExplained Variance Score:\t\t{EVS_4}")


# In[58]:


# Plot Model Predictions (Scatter)
plt.scatter(y_test,predictions_3)

# Plot Perfect predictions (Line)
plt.plot(y_test,y_test,'r')


# ### Model Comparison

# Let us compare the evaluation metrics between models 1, 2, 3, and 4:

# In[75]:


print('EVALUATION METRICS, MODEL 1')
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE}\nRoot Mean Squared Error (RMSE):\t\t{RMSE}\nExplained Variance Score:\t\t{EVS}")
print('-----------------------------\n\n')
print('EVALUATION METRICS, MODEL 2')
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_2}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_2}\nExplained Variance Score:\t\t{EVS_2}")
print('-----------------------------\n\n')
print('EVALUATION METRICS, MODEL 3')
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_3}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_3}\nExplained Variance Score:\t\t{EVS_3}")
print('-----------------------------\n\n')
print('EVALUATION METRICS, MODEL 4')
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_4}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_4}\nExplained Variance Score:\t\t{EVS_4}")


# ## Determining Optimal Number of Hidden Layers

# ### Iterating Through 2-50 Hidden Layers

# We now know that 15 hidden layers is more effective than 20. Let us iterate from 2 to 50 to explore the other possible number of layers, assuming each deep layer contains 8 nodes. Once we find the optimal number of nodes in that range, we can further experiment to optimize the ANN architecture. The for loop below will print out the evaluation metrics for each iteration in real time. Please note that it may take several minutes to run the following code.

# In[138]:


results = []
for i in range(2,51):
    model_loop = Sequential()
    for j in range(0,(i+1)):
        model_loop.add(Dense(8,activation='relu'))
    model_loop.add(Dense(1))

    model_loop.compile(optimizer='adam',loss='mse')
    
    # We will reduce epochs to 200 to reduce run time. 
    # 200 was chosen based on previous loss function visualizations.
    model_loop.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=200,verbose=0)
    
    # Model evaluation
    predictions_loop = model_loop.predict(X_test)

    MAE_loop = mean_absolute_error(y_test,predictions_loop)
    RMSE_loop = np.sqrt(mean_squared_error(y_test,predictions_loop))
    EVS_loop = explained_variance_score(y_test,predictions_loop)

    results.append([i, MAE_loop,RMSE_loop,EVS_loop])
    
    print(f"EVALUATION METRICS, HIDDEN LAYERS = {i}")
    print('-----------------------------')
    print(f"Mean Absolute Error (MAE):\t\t{MAE_loop}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_loop}\nExplained Variance Score:\t\t{EVS_loop}")
    print('-----------------------------\n\n')


# ### Layer Optimization Analysis

# In[144]:


# Convert the results into a numpy array
results_np = np.array(results)

# Store the np array in a pandas dataframe
results_df = pd.DataFrame(columns=['Hidden_Layers','MAE','RMSE','EVS'],data=results_np)


# In[114]:


# Plot the mean absolute error for each iteration
X_plot = results_df['Hidden_Layers']
y_MAE = results_df['MAE']
plt.plot(X_plot,y_MAE)


# In[115]:


# Plot the root mean squared error for each iteration
y_RMSE = results_df['RMSE']
plt.plot(X_plot,y_RMSE)


# In[116]:


# Plot the explained variance score for each iteration
y_EVS = results_df['EVS']
plt.plot(X_plot,y_EVS)


# In[140]:


# Determine the minimum MAE, RMSE, and maximum EVS
results_df.describe()


# In[141]:


# Iteration with the lowest MAE
results_df[results_df['MAE']<4.88]


# In[142]:


# Iteration with the lowest RMSE
results_df[results_df['RMSE']<6.24]


# In[143]:


# Iteration with the largest EVS
results_df[results_df['EVS']>0.85]


# We see that the minimum MAE is from 45 hidden layers, and the lowest RMSE and highest EVS are from 42 hidden layers. We will continue to work with the 44 hidden layer architecture.

# ## Experimenting with the 44 Hidden Layer Model

# There is an infinite number of possible configurations for a neural network. We will explore three below, keeping the 44 total hidden layers.

# ### The Flat Model

# In[162]:


optimization_results = []

model_flat = Sequential()
for i in range(45):
    model_flat.add(Dense(8,activation='relu'))
model_flat.add(Dense(1))

model_flat.compile(optimizer='adam',loss='mse')

# We will reset epochs to 200. 
model_flat.fit(x=X_train,y=y_train.values,
      validation_data=(X_test,y_test.values),
      batch_size=128,epochs=200,verbose=0)

# Model evaluation
predictions_flat = model_flat.predict(X_test)

MAE_flat = mean_absolute_error(y_test,predictions_flat)
RMSE_flat = np.sqrt(mean_squared_error(y_test,predictions_flat))
EVS_flat = explained_variance_score(y_test,predictions_flat)

optimization_results.append(['Flat', MAE_flat,RMSE_flat,EVS_flat])

print(f"EVALUATION METRICS, HIDDEN LAYERS = 44")
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_flat}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_flat}\nExplained Variance Score:\t\t{EVS_flat}")
print('-----------------------------\n\n')


# ### The Descending Model

# In[180]:


model_desc = Sequential()
for i in range(39):
    model_desc.add(Dense(8,activation='relu'))
model_desc.add(Dense(7,activation='relu'))
model_desc.add(Dense(6,activation='relu'))
model_desc.add(Dense(5,activation='relu'))
model_desc.add(Dense(4,activation='relu'))
model_desc.add(Dense(3,activation='relu'))
model_desc.add(Dense(2,activation='relu'))
model_desc.add(Dense(1))

model_desc.compile(optimizer='adam',loss='mse')

model_desc.fit(x=X_train,y=y_train.values,
      validation_data=(X_test,y_test.values),
      batch_size=128,epochs=200,verbose=0)

predictions_desc = model_desc.predict(X_test)

MAE_desc = mean_absolute_error(y_test,predictions_desc)
RMSE_desc = np.sqrt(mean_squared_error(y_test,predictions_desc))
EVS_desc = explained_variance_score(y_test,predictions_desc)

optimization_results.append(['Desc', MAE_desc,RMSE_desc,EVS_desc])

print(f"EVALUATION METRICS, HIDDEN LAYERS = 44, DESCENDING")
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_desc}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_desc}\nExplained Variance Score:\t\t{EVS_desc}")
print('-----------------------------\n\n')


# In[191]:


# We can see that the descencing model performed worse than the flat model

optimization_df = pd.DataFrame(columns=['Model','MAE','RMSE','EVS'],data=np.array(optimization_results))
optimization_df


# ### The Flat Dropout Model

# In[202]:


model_flat_drop = Sequential()

# Input layer
model_flat_drop.add(Dense(8,activation='relu'))

# Hidden Layers
for i in range(22): # Let's make half of the layers in the network dropout layers at a 50% dropout rate
    model_flat_drop.add(Dense(8,activation='relu'))
    
    model_flat_drop.add(Dense(8,activation='relu'))
    model_flat_drop.add(Dropout(0.5))

# Output layer
model_flat_drop.add(Dense(1))

model_flat_drop.compile(optimizer='adam',loss='mse')

model_flat_drop.fit(x=X_train,y=y_train.values,
      validation_data=(X_test,y_test.values),
      batch_size=128,epochs=200,verbose=0)

# Model evaluation
predictions_flat_drop = model_flat_drop.predict(X_test)

MAE_flat_drop = mean_absolute_error(y_test,predictions_flat_drop)
RMSE_flat_drop = np.sqrt(mean_squared_error(y_test,predictions_flat_drop))
EVS_flat_drop = explained_variance_score(y_test,predictions_flat_drop)

optimization_results.append(['Flat_Drop', MAE_flat_drop,RMSE_flat_drop,EVS_flat_drop])

print(f"EVALUATION METRICS, ACTIVE HIDDEN LAYERS = 22, DROPOUT HIDDEN LAYERS = 22")
print('-----------------------------')
print(f"Mean Absolute Error (MAE):\t\t{MAE_flat_drop}\nRoot Mean Squared Error (RMSE):\t\t{RMSE_flat_drop}\nExplained Variance Score:\t\t{EVS_flat_drop}")
print('-----------------------------\n\n')


# ## Conclusions & Recommendations

# We conclude that the "flat model" deep neural network containing 44 hidden layers of 8 nodes each, with no dropout nodes, is the optimal model from all the models tested in this project. 
# 
# Additional models with different numbers of hidden layers and different architectures of the node
# networks could be subject to further experimentation and optimization. All three models containing the 44 hidden layers studied in this project are saved in the Keras_ANN_Models folder. 
# 
# As discussed in the Exploratory Data Analysis code notebook, the compressive strength of concrete inreases rapidly from 0 to 28 days, then more much more stably from 28 days onward. A more intuitive and practical engineering model for predicting the compressive strength of concrete would rely on a given dataset containing only data of a certain curing time. Common testing times are at 3, 7, 14, 28, 60, 90, 128, and 365 days, with the 28 day mark being the industry standard. We analyze linear models at 28 days cure time in the Comparison with Linear Models notebook.
# 
# This dataset presented a unique challenge of predicting compressive strength not only as a function of its constituents, but also of time. The model in this project is able to predict the compressive strength of concrete to within a mean absolute error of 5.08 Megapascals (MPa), a root mean square error of 6.47 MPa, and an explained variance score of 0.838. The actual standard deviation for compressive strength in the dataset is 16.71 MPa. Therefore, the MAE is approximately 0.30σ, and The RMSE is approximately 0.39σ. 
# 
# Given the high variance of the data, particularly in the 0 to 28 day range, these errors are reasonable. We recommend performing additional studies on larger datasets that represent a constant curing time, particularly the standard 28-day curing time, for the most practical engineering applications. Additional analysis comparing the ANN model with linear models is presented in the Model Analysis folder.
