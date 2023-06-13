import numpy as np
import imodels
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
import sklearn.ensemble
import sklearn.tree
import tprompt.tree

class IdentityEnsembleClassifier:
    def __init__(self, n_estimators=1, boosting=False):
        self.n_estimators = n_estimators
        self.boosting = boosting

    def fit(self, X, y):
        if self.boosting:
            self.estimator_ = imodels.BoostedRulesClassifier(
                n_estimators=self.n_estimators
            )
            self.estimator_.fit(X, y)
        else:
            self.estimators_ = []
            for i in range(self.n_estimators):
                clf = DecisionTreeClassifier(max_depth=1)
                clf.fit(X[:, i].reshape(-1, 1), y)
                self.estimators_.append(deepcopy(clf))
        return self

    def predict_proba(self, X):
        # identity ensemble
        if hasattr(self, "estimators_"):
            probas = np.zeros((X.shape[0], self.estimators_[0].n_classes_))
            for i, clf in enumerate(self.estimators_):
                probas += clf.predict_proba(X[:, i].reshape(-1, 1))
            probas /= len(self.estimators_)

        # boosting
        elif hasattr(self, "estimator_"):
            probas = self.estimator_.predict_proba(X)

        return probas

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def _get_model(model_name: str, num_prompts: int, seed: int, args=None):
    if model_name == "tprompt":
        return tprompt.tree.Tree(
            args=args,
            max_depth=args.max_depth,
            split_strategy=args.split_strategy,
            verbose=args.use_verbose,
            checkpoint=args.checkpoint,
            checkpoint_prompting=args.checkpoint_prompting,
        )
    elif model_name == "manual_tree":
        return sklearn.tree.DecisionTreeClassifier(
            max_leaf_nodes=num_prompts + 1,
            random_state=seed,
        )
    elif model_name == "manual_ensemble":
        return IdentityEnsembleClassifier(
            n_estimators=num_prompts,
        )
    elif model_name == "manual_boosting":
        return IdentityEnsembleClassifier(
            n_estimators=num_prompts,
            boosting=True,
        )
    elif model_name == "manual_gbdt":
        return sklearn.ensemble.GradientBoostingClassifier(
            random_state=seed,
        )
    elif model_name == "manual_rf":
        return sklearn.ensemble.RandomForestClassifier(
            random_state=seed,
        )


if __name__ == "__main__":
    X = np.array(
        [
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ]
    ).T
    # y = np.ones(10).astype(int)
    # y[-1] = 0
    # y[3] = 0
    y = np.zeros(10).astype(int)
    y[3:5] = 1
    y[8:10] = 2
    print("shapes", X.shape, y.shape, y)
    for boosting in [False, True]:
        for i in range(X.shape[1]):
            # clf = IdentityEnsembleClassifier(n_estimators=i + 1)
            clf = IdentityEnsembleClassifier(n_estimators=i + 1, boosting=boosting)
            clf.fit(X, y)
            preds = clf.predict(X)
            print(preds)
            print(i, "acc", np.mean(preds == y))
            # print(i, 'clf', clf.estimator_)
