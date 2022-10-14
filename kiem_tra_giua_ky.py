from sklearn.model_selection import train_test_split
import pandas
import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.filterwarnings("ignore")

# Bài 1:
data = pandas.read_csv('data_Ktra2.csv')
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
# Remove NaN
X = X.dropna()
y = data[['Y1']]
# Remove NaN
y = y.dropna()
# Tách dữ liệu thành 2 phần
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3)
# Bài 2
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
X1_test = np.asarray(X_test['X1'])
X2_test = np.asarray(X_test['X2'])
X3_test = np.asarray(X_test['X3'])
X4_test = np.asarray(X_test['X4'])
X5_test = np.asarray(X_test['X5'])
X6_test = np.asarray(X_test['X6'])
y_test = np.asarray(y_test)
# print( 'Solution found by (5): ', w)
# Dự đoán biến Y1
y1 = []
for i in range(X_test.shape[0]):
    y1_pred = w_0 + w_1*X1_test[i] + w_2*X2_test[i] + w_3*X3_test[i] + w_4*X4_test[i] + w_5*X5_test[i] + w_6*X6_test[i]
    y1.append(y1_pred)
# print(y1)
# Sử dụng thư viện
# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(X_train, y_train)
y_pred_train = regr.predict(X_train)
y_pred_val = regr.predict(X_validation)
y_pred_test = regr.predict(X_test)

# (RMSE) Đánh giá mô hình
diff_train = np.sqrt(np.sum(y_train-y_pred_train)**2)/y_train.shape[0]
diff_val = np.sqrt(np.sum(y_validation-y_pred_val)**2)/y_validation.shape[0]
diff_test = np.sqrt(np.sum(y_test-y_pred_test)**2)/y_test.shape[0]
print(diff_train, diff_val, diff_test)
# print( 'Solution found by scikit-learn  : ', regr.coef_)

# Bài 4:
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
bagging_regr = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=30, random_state=0).fit(X_train, y_train.values.ravel())
def baggingPredict(X):
    result = []
    for i in range(X.shape[0]):
        result.append(bagging_regr.predict([X.iloc[i]]))
    return np.asarray(result)
y_pred_train = baggingPredict(X_train)
y_pred_val = baggingPredict(X_validation)
y_pred_test = baggingPredict(X_test)

# (RMSE) Đánh giá mô hình
diff_train = np.sqrt(np.sum(y_train-y_pred_train)**2)/y_train.shape[0]
diff_val = np.sqrt(np.sum(y_validation-y_pred_val)**2)/y_validation.shape[0]
diff_test = np.sqrt(np.sum(y_test-y_pred_test)**2)/y_test.shape[0]
print(diff_train, diff_val, diff_test)