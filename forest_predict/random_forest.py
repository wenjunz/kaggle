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
#train.ix[:,1:11].hist()
train['above_water'] = train.Vertical_Distance_To_Hydrology > 0
train['abs_water'] = abs(train.Vertical_Distance_To_Hydrology)
train['d2h'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
train['f2r_1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['f2r_2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
train['h2r_1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['h2r_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['h2f_1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
train['h2f_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])

test = pd.read_csv('test.csv',header=0)
#test.ix[:,1:11].hist()
test['above_water'] = test.Vertical_Distance_To_Hydrology > 0
test['abs_water'] = abs(test.Vertical_Distance_To_Hydrology)
test['d2h'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
test['f2r_1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['f2r_2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])
test['h2r_1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['h2r_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['h2f_1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
test['h2f_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])

feature_cols = [col for col in train.columns \
        if col not in ['Cover_Type','Id','above_water']]

X_train, X_val, y_train, y_val = train_test_split(\
        train[feature_cols], train['Cover_Type'],test_size=0.5, random_state=0)

#ada = ensemble.AdaBoostClassifier(n_estimators=100, \
#        learning_rate=1.0, algorithm='SAMME.R').fit(X_train,y_train)

forest = ensemble.ExtraTreesClassifier(n_estimators=500, criterion='gini', \
        max_depth=None, min_samples_split=2, min_samples_leaf=1, \
        max_features='auto', bootstrap=True, oob_score=False, n_jobs=-1, \
        random_state=None, verbose=0, min_density=None).fit(X_train,y_train)

mc = confusion_matrix(y_val,forest.predict(X_val))

print('Validation Score is %f' %forest.score(X_val,y_val))

print(pd.DataFrame(forest.feature_importances_,index=feature_cols).sort([0], ascending=False)[:10])

#forest.fit(train[feature_cols],train['Cover_Type'])
#pred = forest.predict(test[feature_cols])

#test['Cover_Type'] = pred;

#test.to_csv('random_forest2.csv', cols=['Id','Cover_Type'],header=True,index=False)


