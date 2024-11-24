# **Project on Predicting House Prices Based on Features**:

# **Objective:**

#The objective of this project is to build a Linear Regression model to predict house prices based on features like square footage, number of bedrooms, and location. The goal is to accurately estimate house prices using the dataset provided.

# **Import Data:**

import pandas as pd

data_url= "https://github.com/YBIFoundation/Dataset/raw/8fc1128e8a4a6e0e01740ebd7ff10da5ee3b4b15/HousePriceLargeDataSet.csv"
data=pd.read_csv("https://github.com/YBIFoundation/Dataset/raw/8fc1128e8a4a6e0e01740ebd7ff10da5ee3b4b15/HousePriceLargeDataSet.csv")
data.head()

# Data Preprocessing

data.info()

data.describe()

data.columns

data.shape

#Handling Missing Data: Handle missing values by either dropping or imputting them with a mean/median.



import numpy as np

# Select only numeric columns for median calculation
numeric_data = data.select_dtypes(include=np.number)

# Calculate the median for numeric columns only
median_values = numeric_data.median()

# Fill NaN values in the original DataFrame using the calculated medians
data.fillna(median_values, inplace=True)

#Feature Engineering: Convert the Categorical Variables into numerical values using encoding methods like one-hot encoding.

data= pd.get_dummies(data,drop_first=True)

#Feature Scalling: Standardize or normalize numerical features if needed.

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data_scaled = scaler.fit_transform(data)

# **Define Target (y) & Feature (x):**

from sklearn.model_selection import train_test_split
x=data.drop('SalePrice',axis=1)
y=data['SalePrice']

x

y

# **Train Test Split**

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

# **Modeling: Select Model**

from sklearn.linear_model import LinearRegression
model=LinearRegression()

#Fit the model: Fit the model to the training data


model.fit(x_train,y_train)

# Prediction

#Predict the house price on the test data

y_pred = model.predict(x_test)

y_pred

#Model Coefficients: Print the coefficients (weights) assigned to each feature.

print("Coefficients: ",model.coef_)

model.intercept_

# Evaluation:

#Mean Absolute Error(MAE): Measure the average magnitude of errors in prediction

#Mean Absolute Percentage Error (MAPE): Measures the average percentage error between the predicted and actual values. Lower MAPE values indicate better model performance.

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error: ",mae)

from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_test,y_pred)
print("Mean Absolute Pecentage Error: ",mape)

#Mean Squared Error(MSE): Measure the average squared difference between actual and predicted values.

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error: ",mse)

#Root Mean Squared Error (RMSE): Provide a estimate of magnitude of error.

import numpy as np
rmse = np.sqrt(mse)
print("Root Mean Squared Error: ",rmse)

#R-squared(R^2): Represents the proportion of variance in the target variable thet is predictable from the independent variables.

r2 = model.score(x_test,y_test)
print("R_sqaured: ",r2)

#**Feature Importance (Coefficients)**

#This graph shows how much each feature contributes to house prices.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming `model.coef_` contains coefficients and `X.columns` contains feature names
coefficients = pd.Series(model.coef_, index=x.columns)
coefficients_sorted = coefficients.sort_values()

plt.figure(figsize=(20, 40))
coefficients_sorted.plot(kind='barh', color='skyblue')
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()


#**Distribution of House Prices**

#Visualizing the distribution of actual house prices.

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.hist(y, bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of House Prices')
plt.xlabel('House Prices')
plt.ylabel('Frequency')
plt.show()


#**Actual vs. Predicted Prices**

#A scatter plot to compare actual house prices against predicted prices.

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.axline([0, 0], [1, 1], color='red', linestyle='--', label='Ideal Fit')
plt.legend()
plt.show()


# **Conclution & Accuracy Score:**

#Adding the R2 score provides a clear understanding of how well the model fits the data. For instance, an R2 score of 0.68 means the model explains 68% of the variance in house prices, indicating strong predictive power.Lower RMSE, MAPE and MAE values fuether validate the model's accuracy.

# **Key Features:**

#*   Larger square footage and more bedrooms drive higher prices.
#*   Older houses or those in less desirable locations may lower prices.

