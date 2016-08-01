#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC,SVR
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.datasets import load_iris

if __name__=="__main__":  
    #X,t=load_iris()["data"],load_iris()["target"]
    X=np.random.uniform(0,10,size=[100,2])
    X = X.reshape(*X.shape)
    t=np.sin(X[:,0]+X[:,1])
    x_train, x_test, t_train, t_test = cross_validation.train_test_split(X, t, test_size=0.2)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma':np.linspace(0.1,3,10,endpoint=True), 'C': np.linspace(0.1,3,10,endpoint=True)}]
    gscv = GridSearchCV(SVR(), tuned_parameters, cv=8, scoring="mean_squared_error")
    gscv.fit(x_train, t_train)
    reg_max = gscv.best_estimator_
    print reg_max
    print reg_max.score(x_test,t_test)
    