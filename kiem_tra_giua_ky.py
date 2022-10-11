from sklearn.model_selection import train_test_split
import pandas
import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import BaggingRegressor

data = pandas.read_csv('data_Ktra2.csv')
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
# Remove NaN
X = X.dropna()
y = data[['Y1']]
# Remove NaN
y = y.dropna()
# Tách dữ liệu thành 2 phần
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = None, test_size = 0.3)

# Xây dựng mô hình hồi quy để dự đoán biến Y1
# Building Xbar
one = np.ones((X_train.shape[0], 1))
Xbar = np.concatenate((one, X_train), axis = 1)
# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y_train)
w = np.dot(np.linalg.pinv(A), b)
# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
X1 = 0.64
X2 = 784
X3 = 343
X4 = 220.5
X5 = 3.5
X6 = 2
y1 = w_0 + w_1*X1 + w_2*X2 + w_3*X3 + w_4*X4 + w_5*X5 + w_6*X6
# print( 'Solution found by (5): ', w)
print(y1)

# Sử dụng thư viện
# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(X_train, y_train)
y_pred1 = regr.predict([[X1,X2,X3,X4,X5,X6]])
print(y_pred1)
# print( 'Solution found by scikit-learn  : ', regr.coef_)

# Bài 4:
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
regr = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=30, random_state=0).fit(X_train, y_train.values.ravel())
y_pred2 = regr.predict([[X1,X2,X3,X4,X5,X6]])
print(y_pred2.T)
