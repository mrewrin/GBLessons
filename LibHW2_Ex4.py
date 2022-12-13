import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('fivethirtyeight')
creditcard = pd.read_csv('creditcard.csv')
class_list = creditcard['Class'].value_counts()
print(class_list)
class_list.plot(kind='barh', logx=True)
plt.show()

class0 = creditcard.loc[creditcard['Class'] == 0, ['V1']]
class1 = creditcard.loc[creditcard['Class'] == 1, ['V1']]
plt.hist(class0['V1'], bins=20, density=True, alpha=0.5, label='Class 0', color='grey')
plt.hist(class1['V1'], bins=20, density=True, alpha=0.5, label='Class 1', color='red')
plt.legend()
plt.show()