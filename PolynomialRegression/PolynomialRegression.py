# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:14:41 2021

@author: Alec
"""

#Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Import Data
salary = pd.read_csv("Employee_Salary.csv")

#Visualize Data
print(salary.head(10))
print(salary.tail(10))
print(salary.describe())
print(salary.info())
#sns.jointplot(x = 'Years of Experience', y = 'Salary', data = salary)
#sns.lmplot(x = 'Years of Experience', y = 'Salary', data = salary)
#sns.jointplot(x = 'Salary', y = 'Years of Experience', data = salary)
#sns.lmplot(x = 'Salary', y = 'Years of Experience', data = salary)
#sns.pairplot(salary)

X = salary[['Years of Experience']]
Y = salary[['Salary']]

X_train = X
Y_train = Y

#Linear Regression Attempt
regressor = LinearRegression(fit_intercept= True)
regressor.fit(X_train, Y_train)

print('Linear Model Coefficient (m)', regressor.coef_)
print('Linear Model Coefficient (b)', regressor.intercept_)

#plt.scatter(X_train, Y_train, color= 'Red')
#plt.plot(X_train, regressor.predict(X_train))
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#plt.title('Years of Experience vs. Salary (Linear)')

#Polynomial Regression
poly_regressor = PolynomialFeatures(degree= 5)
X_columns = poly_regressor.fit_transform(X_train)

regressor = LinearRegression()
regressor.fit(X_columns, Y_train)

print('Linear Model Coefficient (m)', regressor.coef_)
print('Linear Model Coefficient (b)', regressor.intercept_)

Y_predict = regressor.predict(poly_regressor.fit_transform(X_train))

#plt.scatter(X_train, Y_train, color= 'Gray')
#plt.plot(X_train, Y_predict, color= 'Blue')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#plt.title('Years of Experience vs. Salary (Poly order = 5)')

#Import Data
economy = pd.read_csv('EconomiesOfScale.csv')
print(economy.head(10))
print(economy.tail(10))
print(economy.describe())
print(economy.info())


I = economy[['Number of Units']]
J = economy[['Manufacturing Cost']]

I_train = I
J_train = J

#Polynomial Regression
poly_econ_regressor = PolynomialFeatures(degree = 5)
I_columns = poly_econ_regressor.fit_transform(I_train)

econ_regressor = LinearRegression()
econ_regressor.fit(I_columns, J_train)

print('Linear Model Coefficient (m)', econ_regressor.coef_)
print('Linear Model Coefficient (b)', econ_regressor.intercept_)

J_predict = econ_regressor.predict(poly_econ_regressor.fit_transform(I_train))

plt.scatter(I_train, J_train, color='Green')
plt.plot(I_train, J_predict, 'Red')
plt.xlabel('Number of Units')
plt.ylabel('Manufacturing Cost')
plt.title('Number of Units vs Manufacturing Cost (Poly)')
