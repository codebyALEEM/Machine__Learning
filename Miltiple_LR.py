import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load dataset
dataset = pd.read_csv(r'C:\Users\VICTUS\Desktop\mastering git\Practise git\ML_Model\Multiple_LR\Investment.csv')

# Convert data in Independent and Dependent variable
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# Convert categorical data to numberical data 
x = pd.get_dummies(x,dtype=int)

#Split data for TESTING AND TESTING 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# Training model
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Prediction
y_pred = regressor.predict(x_train)

#Slope
m = regressor.coef_
print(m)

#Constant
c = regressor.intercept_
print(c)

#Add constant to dataset
x = np.append(arr=np.full((50,1),42467).astype(int),values=x,axis=1)

#Feature elimination
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OSL = sm.OLS(endog = y, exog =x_opt).fit()
regressor_OSL.summary()


x_opt = x[:,[0,1,2,3,5]]
regressor_OSL = sm.OLS(endog = y, exog =x_opt).fit()
regressor_OSL.summary()

x_opt = x[:,[0,1,2,3]]
regressor_OSL = sm.OLS( endog = y, exog = x_opt).fit()
regressor_OSL.summary()

x_opt = x[:,[0,1,3]]
regressor_OSL = sm.OLS(endog = y, exog= x_opt).fit()
regressor_OSL.summary()

x_opt = x[:,[0,1]]
regressor_OSL = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OSL.summary()


bias = regressor.score(x_train,y_train)
bias

variance = regressor.score(x_test,y_test)
variance

