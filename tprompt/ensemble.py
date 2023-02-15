import numpy as np


class NaiveEnsembleClassifier:
    def __init__(self, n_estimators=1):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        probs = X[:, :self.n_estimators].mean(axis=1)
        probas = np.vstack((1 - probs, probs)).transpose()
        # print('probas.shape', probas.shape)
        return probas

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


if __name__ == '__main__':
    X = np.array(
        [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         ]).T
    y = np.ones(10).astype(int)
    y[-1] = 0
    print('shapes', X.shape, y.shape, y)
    for i in range(X.shape[1]):
        clf = NaiveEnsembleClassifier(n_estimators=i + 1)
        clf.fit(X, y)
        preds = clf.predict(X)
        print(preds)
        print('acc', np.mean(preds == y))
