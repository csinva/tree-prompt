import numpy as np
import imodels

class IdentityEnsembleClassifier:
    def __init__(self, n_estimators=1, boosting=False):
        self.n_estimators = n_estimators
        self.boosting = boosting

    def fit(self, X, y):
        if self.boosting:
            self.estimator_ = imodels.BoostedRulesClassifier(n_estimators=self.n_estimators)
            self.estimator_.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.boosting:
            probas = self.estimator_.predict_proba(X)
        else:
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
    y[3] = 0
    print('shapes', X.shape, y.shape, y)
    for i in range(X.shape[1]):
        # clf = IdentityEnsembleClassifier(n_estimators=i + 1)
        clf = IdentityEnsembleClassifier(n_estimators=i + 1, boosting=True)
        clf.fit(X, y)
        preds = clf.predict(X)
        print(preds)
        print(i, 'acc', np.mean(preds == y))
        print(i, 'clf', clf.estimator_)
