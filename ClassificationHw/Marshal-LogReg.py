# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:12:21 2020

@author: mptay
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


cc_data = pd.read_csv("C:/Users/mptay/Desktop/420/Ensemble/cc.csv")
x=np.array(cc_data.iloc[:,1:len(cc_data.columns)-1])
y=np.array(cc_data['default payment next month'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

cc_model = LogisticRegression().fit(x_train, y_train)

r_sq = cc_model.score(x, y)
print('coefficient of determination:', r_sq)


score = cc_model.score(x_test, y_test)
print(score)

predictions = cc_model.predict(x_test)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

TruePositiveRate = cm[0,0]/ (cm[0,0] + cm[1,0])
TrueNegativeRate = cm[1,1]/ (cm[1,1] + cm[0,1])
Precision = cm[0,0] / (cm[0,0] + cm[0,1])

F1 = (1/2) / ( (1/TruePositiveRate) + (1/Precision) )

import scikitplot as skplt
import matplotlib.pyplot as plt

y_true = y_test
y_probas = cc_model.predict_proba(x_test)
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()
