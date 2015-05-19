import numpy as np
import pandas as pd

import pdb

from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition.pca import PCA

import sklearn.preprocessing

train = pd.read_csv('train.csv',header=0)
train['datetime'] = pd.to_datetime(train['datetime'])
train['year'] = pd.Series([train['datetime'][i].year for i in train.index], index=train.index)-2011
train['month'] = pd.Series([train['datetime'][i].month for i in train.index], index=train.index)
train['day'] = pd.Series([train['datetime'][i].day for i in train.index], index=train.index)
train['hour'] = pd.Series([train['datetime'][i].hour for i in train.index], index=train.index)
train['log_casual'] = np.log(1+train['casual'])
train['log_registered'] = np.log(1+train['registered'])

test = pd.read_csv('test.csv',header=0)
test['datetime'] = pd.to_datetime(test['datetime'])
test['year'] = pd.Series([test['datetime'][i].year for i in test.index], index=test.index)-2011
test['month'] = pd.Series([test['datetime'][i].month for i in test.index], index=test.index)
test['day'] = pd.Series([test['datetime'][i].day for i in test.index], index=test.index)
test['hour'] = pd.Series([test['datetime'][i].hour for i in test.index], index=test.index)

feature_cols = [col for col in train.columns \
        if col not in ['datetime','casual','registered','count','log_casual','log_registered']]

X_train_casual, X_val_casual, y_train_casual, y_val_casual = train_test_split(\
        train[feature_cols], train['log_casual'],test_size=0.5, random_state=0)

forest_casual = ensemble.RandomForestRegressor(n_estimators=100, criterion='mse', \
        max_depth=None, min_samples_split=11, min_samples_leaf=1, \
        max_features='auto', bootstrap=True, oob_score=False, n_jobs=-1, \
        random_state=None, verbose=0).fit(X_train_casual,y_train_casual)
print('Validation Score is %f' %forest_casual.score(X_val_casual,y_val_casual))

y_val = y_val_casual + y_val_registered;
print('Validation Score is %f' %forest_registered.score(X_val,y_val))
#mc = confusion_matrix(y_val,forest.predict(X_val))

#print('Validation Score is %f' %forest.score(X_val,y_val_casual))

#print(pd.DataFrame(forest.feature_importances_,index=feature_cols).sort([0], ascending=False)[:10])

#forest.fit(train[feature_cols],train['Cover_Type'])
pred_casual = forest_casual.predict(test[feature_cols])
pred_registered = forest_casual.predict(test[feature_cols])

pred_count = map(int,pred_casual+pred_registered)
test['count'] = pred_count;

test.to_csv('random_forest1.csv', cols=['datetime','count'],header=True,index=False)


