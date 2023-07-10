# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:44:50 2022

@author: Sumit Patre
"""

import pandas as pd
import numpy as np  
data = pd.read_csv("C:/Users/user/OneDrive/Desktop/py 2/simple linear/20 Simple Linear Regression_Problem Statement/calories_consumed.csv")
data1=data.rename({"Weight gained (grams)":"weight","Calories Consumed":"calory"},axis=1)
data1.describe()

#Graphical Representation
import matplotlib.pyplot as plt

plt.bar(height = data1.calory, x = np.arange(1, 15, 1))
plt.hist(data1.calory)                    #histogram
plt.boxplot(data1.calory)                 #boxplot

plt.bar(height = data1.weight, x = np.arange(1, 15, 1))
plt.hist(data1.weight)                 #histogram
plt.boxplot(data1.weight)              #boxplot

# Scatter plot
plt.scatter(x = data1['weight'], y = data1['calory'], color = 'green') 

# correlation
np.corrcoef(data1.weight, data1.calory) 

cov_output = np.cov(data1.weight, data1.calory)[0, 1]
cov_output


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('calory ~ weight', data = data1).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(data1['weight']))

# Regression Line
plt.scatter(data1.weight, data1.calory)
plt.plot(data1.weight, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = data1.calory - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(weight); y = calory

plt.scatter(x = np.log(data1['weight']), y = data1['calory'], color = 'brown')
np.corrcoef(np.log(data1.weight), data1.calory) #correlation

model2 = smf.ols('calory ~ np.log(weight)', data = data1).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(data1['weight']))

# Regression Line
plt.scatter(np.log(data1.weight), data1.calory)
plt.plot(np.log(data1.weight), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = data1.calory - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = weight; y = log(calory)

plt.scatter(x = data1['weight'], y = np.log(data1['calory']), color = 'orange')
np.corrcoef(data1.weight, np.log(data1.calory)) #correlation

model3 = smf.ols('np.log(calory) ~ weight', data = data1).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(data1['weight']))
pred3_calory = np.exp(pred3)
pred3_calory

# Regression Line
plt.scatter(data1.weight, np.log(data1.calory))
plt.plot(data1.weight, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = data1.calory - pred3_calory
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = weight; x^2 = weight*weight; y = log(calory)

model4 = smf.ols('np.log(calory) ~ weight + I(weight*weight)', data = data1).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(data1))
pred4_calory = np.exp(pred4)
pred4_calory

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = data1.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = data1.iloc[:, 1].values


plt.scatter(data1.weight, np.log(data1.calory))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = data1.calory - pred4_calory
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(data1, test_size = 0.2)

finalmodel = smf.ols('np.log(calory) ~ weight + I(weight*weight)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_calory = np.exp(test_pred)
pred_test_calory

# Model Evaluation on Test data
test_res = test.calory - pred_test_calory
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_calory = np.exp(train_pred)
pred_train_calory

# Model Evaluation on train data
train_res = train.calory - pred_train_calory
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
