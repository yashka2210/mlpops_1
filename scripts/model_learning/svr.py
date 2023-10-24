import pandas as pd
from sklearn.svm import SVR
import pickle
import os
# SVM

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_squared_error as mse, r2_score

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate

from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV
#from sklearn.utils.fixes import loguniform

epsilon=20 

C=100 

kernel = 'poly' 
coef0=4.5 
degree=3 
gamma=0.4 

if gamma == 0:
  gamma='auto'

SVM_reg=SVR(kernel=kernel, epsilon=epsilon, C=C, gamma=gamma,
            degree=degree, coef0 = coef0)

X_Train = pd.read_csv('data/stage_2/X_Train.csv', index_col = 0)

y_Train = pd.read_csv('data/stage_2/y_Train.csv', index_col = 0)

SVM_reg.fit(X_Train, y_Train)

with open("models", "wb") as fd:
  pickle.dump(SVM_reg, fd)