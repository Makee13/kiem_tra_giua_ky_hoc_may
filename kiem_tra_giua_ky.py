from sklearn.model_selection import train_test_split
import pandas
import numpy as np

data = pandas.read_csv('data_Ktra2.csv')
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
# Remove NaN
X = X.dropna()
y = data[['Y1']]
# Remove NaN
y = y.dropna()
# Tách dữ liệu thành 2 phần
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 50, test_size = 0.3)

# Xây dựng mô hình hồi quy để dự đoán biến Y1
# Building Xbar
one = np.ones((X_train.shape[0], 1))
Xbar = np.concatenate((one, X_train), axis = 1)

# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y_train)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
X1 = 0.62
X2 = 808.5
X3 = 367.5
X4 = 220.5
X5 = 3.5
X6 = 3
y1 = w_0 + w_1*X1 + w_2*X2 + w_3*X3 + w_4*X4 + w_5*X5 + w_6*X6
print(y1)

# Drawing the fitting line
# plt.plot(X.T, y.T, 'ro')     # data
# plt.plot(x0, y0)               # the fitting line
# plt.axis([140, 190, 45, 75])
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
# plt.show()
