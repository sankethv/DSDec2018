import os
import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn import model_selection
import pydot
import io

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("D:/Data Science/Data")

titanic_train = pd.read_csv("titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

dt_estimator = tree.DecisionTreeClassifier()
#Base_estimaor = dt_estimator, n_estimators = 5(no. of Trees to be grown)
ada_tree_estimator1 = ensemble.AdaBoostClassifier(base_estimator = dt_estimator, n_estimators = 5, learning_rate =0.2, random_state=2017)
#==============================================================================
# scores = model_selection.cross_val_score(ada_tree_estimator1, X_train, y_train, cv = 10)
# print(scores.mean())
#==============================================================================
ada_tree_estimator1.fit(X_train, y_train)

#extracting all the trees build by Ada Boost algorithm
n_tree = 0
for est in ada_tree_estimator1.estimators_: 
    dot_data = io.StringIO()
    tmp = est.tree_
    tree.export_graphviz(tmp, out_file = dot_data, feature_names = X_train.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())#[0] 
    graph.write_pdf("adatree" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1
    
    
