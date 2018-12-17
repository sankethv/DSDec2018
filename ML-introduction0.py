# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:20:44 2018

@author: Sreenivas.J
"""

import pandas as pd
import os
from sklearn import tree #For Decissin Tree

#Read Train Data file
titanic_train = pd.read_csv("D:\\Data Science\\Data\\titanic_train.csv")

titanic_train.shape #Not mandatory though!!
titanic_train.info() #Not mandatory though!!

#Let's start the journey with non categorical and non missing data columns
X_titanic_train = titanic_train[['Pclass', 'SibSp', 'Parch']] #X-Axis
y_titanic_train = titanic_train['Survived'] #Y-Axis

#Build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_titanic_train, y_titanic_train)

#Predict the outcome using decision tree
#Read the Test Data
titanic_test = pd.read_csv("D:\\Data Science\\Data\\titanic_test.csv")
X_test = titanic_test[['Pclass', 'SibSp', 'Parch']]
#Use .predict method on Test data using the model which we built
titanic_test['Survived'] = dt.predict(X_test) 
os.getcwd() #To get current working directory
titanic_test.to_csv("submission_Titanic.csv", columns=['PassengerId','Survived'], index=False)
