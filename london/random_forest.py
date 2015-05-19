import numpy as np
import pandas as pd

import pdb

from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import sklearn.preprocessing


train = pd.read_csv('train.csv',header=None)
train_y = pd.read_csv('trainLabels.csv',header=None)
test = pd.read_csv('test.csv',header=None)
#test.ix[:,1:11].hist()

feature_cols = [col for col in train.columns]

X_train, X_val, y_train, y_val = \
        train_test_split(train[feature_cols], train_y,test_size=0.5, random_state=0)

clf = SVC().fit(X_train,y_train)

forest = ensemble.RandomForestClassifier(n_estimators=500, criterion='gini', \
        max_depth=None, min_samples_split=2, min_samples_leaf=1, \
        max_features='auto', bootstrap=False, oob_score=False, n_jobs=-1, \
        random_state=None, verbose=0, min_density=None).fit(X_train,y_train)

mc = confusion_matrix(y_val,forest.predict(X_val))
print('Validation Score is %f' %forest.score(X_val,y_val))
print(pd.DataFrame(forest.feature_importances_,index=feature_cols).sort([0], ascending=False)[:10])

forest.fit(train[feature_cols],train['Cover_Type'])
pred = forest.predict(test[feature_cols])

test['Cover_Type'] = pred;

test.to_csv('random_forest.csv', cols=['Id','Cover_Type'],header=True,index=False)


