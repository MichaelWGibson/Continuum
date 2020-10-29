import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

SEARCH_DT = False
SEARCH_FOREST = False
SEARCH_BAGGING = False
SEARCH_ADA = False
SEARCH_GB = False

DT = True
FOREST = True
BAGGING = True
ADA = True
GB = True

data = pd.read_csv('data.csv')

X = data.iloc[:,:24]
Y = data['default payment next month']

#Split the data into training and testing sets.
X_train_org, X_test_org, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.2)

scaler = MinMaxScaler() #Scaling the data because sometimes, the data varies a lot(check X.describe())
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.fit_transform(X_test_org)

final_performances = []
final_algs = []

if SEARCH_DT:
    param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                  'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'criterion': ['entropy', 'gini']}

    print("Parameter grid:\n{}".format(param_grid))

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 6, return_train_score = True)
    grid_search.fit(X_train, y_train)

    print("Best score: {:.4f}".format(grid_search.best_score_))
    print("Best parameters: {}".format(grid_search.best_params_))

if DT:
    bestdt = DecisionTreeClassifier(criterion = "gini", max_depth = 3, min_samples_leaf = 1)
    bestdt.fit(X_train, y_train)
    print("Decision Tree")
    print("Training Accuracy:",bestdt.score(X_train, y_train)) #Accuracy of the model when training.
    print("Testing Accuracy:", bestdt.score(X_test, y_test) )#Accuracy of the test.
    final_performances.append(bestdt.score(X_test,y_test))
    final_algs.append("Decision Tree")

if SEARCH_FOREST:
    param_grid = {'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                  'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'random_state':[42]}
    print("parameter grid:\n{}".format(param_grid))


    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv = 6, return_train_score = True)
    grid_search.fit(X_train, y_train)

    print("Forest Search")
    print("Best score: {:.4f}".format(grid_search.best_score_))
    print("Best parameters: {}".format(grid_search.best_params_))

if SEARCH_BAGGING:
    param_grid = {'n_estimators': [100, 250, 500, 750, 1000],
                  'base_estimator': [bestdt],
                  'random_state':[42]}
    print("parameter grid:\n{}".format(param_grid))

    grid_search = GridSearchCV(BaggingClassifier(), param_grid, cv = 6, return_train_score = True)
    grid_search.fit(X_train, y_train)

    print("Bagging Search")
    print("Best score: {:.4f}".format(grid_search.best_score_))
    print("Best parameters: {}".format(grid_search.best_params_))

if SEARCH_ADA:
    param_grid = {'n_estimators': [100, 250, 500, 750, 1000],
                  'learning_rate': [.1, .5, 1, 3, 4, 5],
                  'base_estimator': [bestdt],
                  'random_state':[42]}
    print("parameter grid:\n{}".format(param_grid))

    grid_search = GridSearchCV(AdaBoostClassifier(), param_grid, cv = 6, return_train_score = True)
    grid_search.fit(X_train, y_train)

    print("ADA Search")
    print("Best score: {:.4f}".format(grid_search.best_score_))
    print("Best parameters: {}".format(grid_search.best_params_))

if SEARCH_GB:
    param_grid = {'n_estimators': [100, 250, 500, 750, 1000],
                  'learning_rate': [.1, .5, 1, 3, 4, 5],
                  'random_state':[42]}
    print("parameter grid:\n{}".format(param_grid))

    grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv = 6, return_train_score = True)
    grid_search.fit(X_train, y_train)

    print("GB Search")
    print("Best score: {:.4f}".format(grid_search.best_score_))
    print("Best parameters: {}".format(grid_search.best_params_))

if FOREST:
    rf = RandomForestClassifier(n_estimators = 14, random_state = 42, max_depth = 8, min_samples_leaf = 5)
    rf.fit(X_train, y_train)

    print("Random Forest")
    print("Training accuracy:",rf.score(X_train, y_train)) #accuracy of the model when training.
    print("Testing accuracy:", rf.score(X_test, y_test) )#accuracy of the test.
    final_performances.append(rf.score(X_test,y_test))
    final_algs.append("Random Forest")

if BAGGING:
    bag = BaggingClassifier(base_estimator = bestdt, n_estimators=500)
    bag.fit(X_train, y_train)

    print("Bagging")
    print("Training Accuracy:",bag.score(X_train, y_train)) #Accuracy of the model when training.
    print("Testing Accuracy:", bag.score(X_test, y_test) )#Accuracy of the test.
    final_performances.append(bag.score(X_test,y_test))
    final_algs.append("Bagging")

if ADA:
    ada = AdaBoostClassifier(base_estimator = bestdt, learning_rate=0.1, n_estimators=100, random_state=42)
    ada.fit(X_train, y_train)

    print("ADA")
    print("Training Accuracy:",ada.score(X_train, y_train)) #Accuracy of the model when training.
    print("Testing Accuracy:", ada.score(X_test, y_test) )#Accuracy of the test.
    final_performances.append(ada.score(X_test,y_test))
    final_algs.append("ADA")

if GB:
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)

    print("Gradient Boost")
    print("Training Accuracy:",gb.score(X_train, y_train)) #Accuracy of the model when training.
    print("Testing Accuracy:", gb.score(X_test, y_test) )#Accuracy of the test.
    final_performances.append(gb.score(X_test,y_test))
    final_algs.append("Gradient Boost")

print(final_performances)
print(final_algs)
