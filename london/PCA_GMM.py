import numpy as np
import pandas as pd

import sklearn.preprocessing
from sklearn.svm import SVC
from sklearn.mixture import GMM
from sklearn.decomposition.pca import PCA
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import pdb

train = pd.read_csv('train.csv',header=None)
train_y = pd.read_csv('trainLabels.csv',header=None)
test = pd.read_csv('test.csv',header=None)
#test.ix[:,1:11].hist()

n_pca = 21;
n_gmm = 4;

pca = PCA(n_components=n_pca,whiten=True).fit(train)
train_pca = pca.transform(train)

X_train, X_val, y_train, y_val = \
        train_test_split(train_pca, train_y,test_size=0.2, random_state=0)

gmm = GMM(n_components=n_gmm,covariance_type='full').fit(X_train)
svc = SVC().fit(gmm.predict_proba(X_train),y_train)
svc.score(gmm.predict_proba(X_val),y_val)

forest = ensemble.ExtraTreesClassifier(n_estimators=400).fit(gmm.predict_proba(X_train),y_train)
forest.score(gmm.predict_proba(X_val),y_val)

test_pca = pca.transform(test)
gmm_all = GMM(n_components=n_gmm,covariance_type='full').fit(train_pca)

svc_all = SVC().fit(gmm_all.predict_proba(train_pca),train_y)
pred_svc = svc_all.predict(gmm_all.predict_proba(test_pca))

forest_all = ensemble.RandomForestClassifier(n_estimators=400).fit(gmm_all.predict_proba(train_pca),train_y)
pred_forest = forest_all.predict(gmm_all.predict_proba(test_pca))

submission = pd.DataFrame({'Id':np.arange(1,9001),'solution':pred_forest})

submission.to_csv('PCA_GMM_4.csv', cols=['Id','solution'],header=True,index=False)

def kde_plot(x):
    from scipy.stats.kde import gaussian_kde
    kde = gaussian_kde(x)
    positions = np.linspace(x.min(), x.max())
    smoothed = kde(positions)
    plt.plot(positions, smoothed)

def qq_plot(x):
    from scipy.stats import probplot
    probplot(x, dist='norm', plot=plt)
        

