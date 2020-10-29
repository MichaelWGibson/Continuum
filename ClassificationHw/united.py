import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, matthews_corrcoef
import scikitplot as skplt
import matplotlib as plt

data = pd.read_csv('data.csv')

X = data.iloc[:,:24]
Y = data['default payment next month']

#Split the data into training and testing sets.
X_train_org, X_test_org, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.2)

scaler = MinMaxScaler() #Scaling the data because sometimes, the data varies a lot(check X.describe())
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.fit_transform(X_test_org)




# Decision Tree
print("Fitting Decision Tree")
dt = DecisionTreeClassifier(criterion = "gini", max_depth = 3, min_samples_leaf = 1)
dt.fit(X_train, y_train)
print("Training Accuracy for DT:",dt.score(X_train, y_train)) # Accuracy of the model when training.
print("Testing Accuracy for DT:", dt.score(X_test, y_test) ) # Accuracy of the test.

# K-Nearest-Neighbor
print("Fitting KNN")
knn = KNeighborsClassifier(15)
knn.fit(X_train, y_train)
print("Training Accuracy for DT:",knn.score(X_train, y_train)) # Accuracy of the model when training.
print("Testing Accuracy for DT:", knn.score(X_test, y_test) ) # Accuracy of the test.

# Stochastic Gradient Descent
print("Fitting SGD")
sgd = SGDClassifier(max_iter=1690, penalty='l1', early_stopping=False)
sgd.fit(X_train, y_train)
print("Training Accuracy for SGD:",sgd.score(X_train, y_train)) # Accuracy of the model when training.
print("Testing Accuracy for SGD:", sgd.score(X_test, y_test) ) # Accuracy of the test.

# Support Vector Classifier
print("Fitting SVC")
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
print("Training Accuracy for SVC:",svc.score(X_train, y_train)) # Accuracy of the model when training.
print("Testing Accuracy for SVC:", svc.score(X_test, y_test) ) # Accuracy of the test.

# Logistic Regression Classifier
print("Fitting Logistic Regression")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print("Training Accuracy for LR:",lr.score(X_train, y_train)) # Accuracy of the model when training.
print("Testing Accuracy for LR:", lr.score(X_test, y_test) ) # Accuracy of the test.




# Final Voting Classifier
print("Fitting final Voting Classifier")
vc = VotingClassifier(estimators=[('dt', dt), ('lr', lr), ('knn', knn), ('svc', svc), ('sgd', sgd)], voting='hard')
vc.fit(X_train, y_train)

print("Training Accuracy:",vc.score(X_train, y_train)) # Accuracy of the model when training.
print("Testing Accuracy:", vc.score(X_test, y_test) ) # Accuracy of the test.

predictions = vc.predict(X_test)
cm = confusion_matrix(y_test, predictions)
print(cm)

print( "F1: ", f1_score(y_test, predictions))
print( "Recall: ", recall_score(y_test, predictions, average='micro'))
print( "Accuracy: ", accuracy_score(y_test, predictions))
print( "Matthews Correlation Coefficient", matthews_corrcoef(y_test, predictions))

y_probas = [c.predict_proba(X_test) for c in (dt, lr, knn, svc, sgd)]
skplt.metrics.plot_roc(y_test, y_probas)
plt.show()
plt.savefig("roc_curve.png")
