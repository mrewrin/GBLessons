import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

clf = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
#print(clf)
clf.fit(X_train, Y_train.values[:, 0])
#print(clf.fit())
y_pred_clf = clf.predict(X_test)
check_test_clf = pd.DataFrame({
    "Y_test": Y_test["price"],
    "Y_pred_clf": y_pred_clf.flatten()})
check_test_clf.head()
#print(check_test_clf.head())

mean_squared_error_clf = mean_squared_error(check_test_clf["Y_pred_clf"], check_test_clf["Y_test"])
print( mean_squared_error_clf)

#RandomForest работает лучше, более точно.

print(clf.feature_importances_)
feature_importance = pd.DataFrame({'name':X.columns,
                                   'feature_importance':clf.feature_importances_},
                                  columns=['feature_importance', 'name'])
feature_importance

feature_importance.nlargest(2, 'feature_importance')