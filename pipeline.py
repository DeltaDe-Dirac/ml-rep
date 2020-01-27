# 1. import a dataset
from sklearn import datasets

# 2. prepare T&T data
from sklearn.model_selection import train_test_split

# 3. prepare classifier
from sklearn.neighbors import KNeighborsClassifier

# 4. metrics and visualization
from sklearn.metrics import accuracy_score

# 1.
iris = datasets.load_iris()

X = iris.data
y = iris.target

# 2.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# 3.
neighbor_classifier = KNeighborsClassifier()
neighbor_classifier.fit(X_train, y_train)

predictions = neighbor_classifier.predict(X_test)

# 4.
print(accuracy_score(y_test, predictions))
