# import a dataset
from sklearn import datasets
# prepare T&T data
from sklearn.model_selection import train_test_split
# build classifier class
from scipy.spatial import distance
# metrics and visualization
from sklearn.metrics import accuracy_score


class ScrappyKNN:

    def __init__(self):
        self.out_train = []
        self.in_train = []
        self.predictions = []

    def fit(self, in_train, out_train):
        self.out_train = out_train
        self.in_train = in_train

    def predict(self, in_test):
        self.predictions.clear()

        for row in in_test:
            label = self.closest(row)
            self.predictions.append(label)

        return self.predictions

    def closest(self, row):
        shortest_dist = ScrappyKNN.euc(row, self.in_train[0])
        shortest_idx = 0

        for i in range(1, len(self.in_train)):
            dist = ScrappyKNN.euc(row, self.in_train[i])
            if dist < shortest_dist:
                shortest_dist = dist
                shortest_idx = i
        return self.out_train[shortest_idx]

    @staticmethod
    def euc(a, b):
        return distance.euclidean(a, b)


# ------------------------------- MAIN ------------------------------- #
if __name__ == '__main__':
    # 1.
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    # 2.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

    # 3.
    scrappy_clf = ScrappyKNN()
    scrappy_clf.fit(X_train, y_train)

    predictions = scrappy_clf.predict(X_test)

    # 4.
    print(accuracy_score(y_test, predictions))
# ------------------------------- MAIN ------------------------------- #
