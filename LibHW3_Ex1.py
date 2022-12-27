import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

california = fetch_california_housing()
data = california["data"]
feature_names = california["feature_names"]

X = pd.DataFrame(data, columns=feature_names)
X.head()
#print(X)
target = california["target"]
Y = pd.DataFrame(target, columns=["price"])
Y.head()
#print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred_lr = lr.predict(X_test)
check_test_lr = pd.DataFrame({
    "Y_test": Y_test["price"],
    "Y_pred_lr": y_pred_lr.flatten()})
check_test_lr.head()
#print(check_test_lr.head)

mean_squared_error_lr = mean_squared_error(check_test_lr["Y_pred_lr"], check_test_lr["Y_test"])
print(mean_squared_error_lr)