# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 15:13:32 2018

@author: wdz
"""
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import time

iris = datasets.load_iris()

X = iris.data
y = iris.target


in_sample_score = []
out_sample_score = []

start_1 = time.time()


for i in range(1,11):
    tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)
    X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.1,
                                                         random_state = i, stratify = y)
    classifier = tree
    # f-string is a 3.6 thing but I like it: it is way more clear for understanding
    #e.g f'{a:.2f}'
    print(f'using random state {i}')
    
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    print(f'train accuracy {accuracy_score(y_train, y_train_pred) :.4f}')
    y_test_pred = classifier.predict(X_test)
    print(f'test accuracy {accuracy_score(y_test, y_test_pred) :.4f}')
    in_sample_score.append(accuracy_score(y_train, y_train_pred))
    out_sample_score.append(accuracy_score(y_test, y_test_pred))

in_sam_mean = np.mean(in_sample_score)
in_sam_std = np.std(in_sample_score)
out_sam_mean = np.mean(out_sample_score)
out_sam_std = np.std(out_sample_score)

end_1 = time.time()
random_state_time = end_1 - start_1
print(f'loop thru random state time is {random_state_time}')
print(f'In sample mean is {in_sam_mean :.4f}, in sample standard deviation is {in_sam_std :.4f}')
print(f'Out sample mean is {out_sam_mean :.4f}, out sample standard deviation is {out_sam_std :.4f}')



start_2 = time.time()

X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.1,
                                                         random_state = 1, stratify = y)
tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)
cv_scores = cross_val_score(estimator = tree, X = X_train, y = y_train, cv = 10, n_jobs = 1)
print(f'CV accuracy scores {cv_scores}')
print(f'CV accuracy: {np.mean(cv_scores)} +/- {np.std(cv_scores)}')

end_2 = time.time()
# so far our task 2 ends
param_range = {'random_state': range(1,11,1)}
gs = GridSearchCV(estimator = tree, param_grid = param_range, scoring = 'accuracy',
                  cv = 10)
gs.fit(X_train, y_train)
print(f'best param is {gs.best_params_}, best score is {gs.best_score_}')

k_fold_cv_time = end_2 - start_2
print(f'k fold cv time is {k_fold_cv_time}')


time_delta = random_state_time - k_fold_cv_time
print(f'random_state run time is larger than k fold cv run time by {time_delta}')
if (time_delta) > 0:
    print('cv takes less computation time')
else:
    print('random state takes less computation time')


print("My name is Dizhou Wu")
print("My NetID is: dizhouw2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")