# STEP 1 imports
from sklearn.datasets import load_iris
# STEP 2 imports
import numpy as np
# STEP 3 imports
from sklearn import tree
# STEP 4 imports
from six import StringIO
import pydot
# Utils imports
import os

# ----- STEP 1 ----- #
# Loading featured data
iris = load_iris()

# Set of all features describing given data set for prediction:
# 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
print(iris.feature_names)

# Available predictions as flower names:
# 0 - 'setosa' 1 - 'versicolor' 2 - 'virginica'
print(iris.target_names)

# Example: setosa features
print(iris.data[0])
print(iris.target[0])

# Example: versicolor features
print(iris.data[50])
print(iris.target[50])

# Example: virginica features
print(iris.data[100])
print(iris.target[100])

# All available data
for i in range(len(iris.target)):
    print("Example %d: LABEL [%s] ... FEATURES %s" % (i, iris.target[i], iris.data[i]))
# ----- STEP 1 ----- #


# =======================================================================================


# ----- STEP 2 ----- #
# Creating Training and Testing data
test_idx = [0, 50, 100]

# Removing 3 elements from training data for testing purposes
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing data:
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
# ----- STEP 2 ----- #


# =======================================================================================


# ----- STEP 3 ----- #
# Creating and training the classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print("Expected: %s" % test_target)
print("Predicted: %s" % clf.predict(test_data))
# ----- STEP 3 ----- #


# =======================================================================================


# ----- STEP 4 ----- #
# Visualization with viz code

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,
                     rounded=True,
                     impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write("Docs/iris.pdf", format='pdf')

os.system('cd Docs & iris.pdf')
# ----- STEP 4 ----- #


# =======================================================================================
