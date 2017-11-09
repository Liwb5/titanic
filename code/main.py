import pandas as pd
import numpy as np
import matplotlib as plt

import xgboost as xgb
from sklearn import preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

import dataProcess as dp

trainPath = '../data/train.csv'
testPath = '../data/test.csv'
x_train, y_train,x_test = dp.loadData(trainPath,testPath)

xgb_model = xgb.XGBClassifier()

parameters = {'nthread': [4],
             'objective': ['binary:logistic'],
             'learning_rate': [0.01,0.03,0.05],
             'max_depth': [4, 5, 6],
             'min_child_weight': [11],
             'silent': [1],
             'subsample': [0.8],
             'colsample_bytree': [0.7],
             'n_estimators': [2000],
             'seed': [1337],}
clf = GridSearchCV(xgb_model, param_grid=parameters, n_jobs=5,
                  cv=StratifiedKFold(y_train, n_folds=10, shuffle=True),
                  scoring='accuracy', verbose=2, refit=True)

clf.fit(x_train, y_train)

best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('accuracy_score ', score)

for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
