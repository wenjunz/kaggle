import numpy as np
import pandas as pd

import pdb

from sklearn.ensemble import RandomForestRegressor
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
train['weekday'] = pd.Series([train['datetime'][i].weekday() for i in train.index], index=train.index)
train['log_casual'] = np.log(1+train['casual'])
train['log_registered'] = np.log(1+train['registered'])

test = pd.read_csv('test.csv',header=0)
test['datetime'] = pd.to_datetime(test['datetime'])
test['year'] = pd.Series([test['datetime'][i].year for i in test.index], index=test.index)-2011
test['month'] = pd.Series([test['datetime'][i].month for i in test.index], index=test.index)
test['day'] = pd.Series([test['datetime'][i].day for i in test.index], index=test.index)
test['hour'] = pd.Series([test['datetime'][i].hour for i in test.index], index=test.index)
test['weekday'] = pd.Series([test['datetime'][i].weekday() for i in test.index], index=test.index)

feature_cols = [col for col in train.columns \
        if col not in ['year','month','day','datetime','casual','registered','count','log_casual','log_registered']]

#parameters = {'n_estimators':[100,500,1000,2000],\
#        'min_samples_split':[1,5,10,20]}

# casual count
#clf_casual = GridSearchCV(RandomForestRegressor(),parameters,n_jobs=2)
#clf_casual.fit(train[feature_cols],train['log_casual'])
#print(pd.DataFrame(clf_casual.best_estimator_.feature_importances_,index=feature_cols).sort([0], ascending=False)[:10])
#cm_casual = confusion_matrix(train['log_casual'],clf_casual.predict(train[feature_cols]))
#forest_casual = clf_casual.best_estimator_.fit(train[feature_cols],train['log_casual'])
forest_casual = RandomForestRegressor(n_estimators=1000,min_samples_split=11).fit(train[feature_cols],train['log_casual'])
print(pd.DataFrame(forest_casual.feature_importances_,index=feature_cols).sort([0], ascending=False)[:10])
pred_casual = np.round(np.exp(forest_casual.predict(test[feature_cols]))-1)

# registered count
#clf_registered = GridSearchCV(RandomForestRegressor(),parameters,n_jobs=2)
#clf_registered.fit(train[feature_cols],train['log_registered'])
#print(pd.DataFrame(clf_registered.best_estimator_.feature_importances_,index=feature_cols).sort([0], ascending=False)[:10])
#cm_registered = confusion_matrix(train['log_registered'],clf_registered.predict(train[feature_cols]))
#forest_registered = clf_registered.best_estimator_.fit(train[feature_cols],train['log_registered'])
forest_registered = RandomForestRegressor(n_estimators=1000,min_samples_split=11).fit(train[feature_cols],train['log_registered'])
print(pd.DataFrame(forest_registered.feature_importances_,index=feature_cols).sort([0], ascending=False)[:10])
pred_registered = np.round(np.exp(forest_registered.predict(test[feature_cols]))-1)


pred_count = pred_casual+pred_registered
test['count'] = pred_count;
test.to_csv('random_forest6.csv', cols=['datetime','count'],header=True,index=False)


