import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn.externals import joblib #For exporting and importing

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("D:\\Sanketh\\Data Science\\Titanic")

titanic_train = pd.read_csv("titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#data preparation
titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

#feature engineering 
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#build the decision tree model
dt = tree.DecisionTreeClassifier(random_state=2017)

dt_grid = {'criterion':['gini','entropy'], 'max_depth':list(range(3,12)), 'min_samples_split':[2,3,6,7,8]}
param_grid = model_selection.GridSearchCV(dt, dt_grid, cv=20) #Evolution of tee
param_grid.fit(X_train, y_train) #Building the tree
print(param_grid.best_score_) #Best score
print(param_grid.best_params_)
print(param_grid.score(X_train, y_train)) #train score  #Evolution of tree

# natively deploy decision tree model(pickle format)
os.getcwd()
joblib.dump(param_grid, "TitanicVer1.pkl")
