from typing import List
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import imodels
import imodelsx.util
import imodelsx.metrics
import logging
from abc import ABC, abstractmethod


STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'nor', 'only', 'own', 'so', 'than', 'too', 'very',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

class Stump(ABC):
    def __init__(
        self,
        split_strategy: str='iprompt',
        tokenizer=None,
        max_features=10,
        assert_checks: bool=False,
        verbose: bool=True,
    ):
        """Fit a single stump.
        Can use tabular features...
            Currently only supports binary classification with binary features.
        Params
        ------
        split_strategy: str
            'iprompt' - use iprompt to split
            'cart' - use cart to split
            'linear' - use linear to split
        max_features: int
            used by StumpTabular to decide how many features to save
        """
        assert split_strategy in ['iprompt', 'cart', 'linear']
        self.split_strategy = split_strategy
        self.assert_checks = assert_checks
        self.verbose = verbose
        self.max_features = max_features 
        if tokenizer is None:
            self.tokenizer = imodelsx.util.get_spacy_tokenizer(convert_output=False)
        else:
            self.tokenizer = tokenizer

        # tree stuff
        self.child_left = None
        self.child_right = None
    
    @abstractmethod
    def fit(self, X, y, feature_names=None, X_text: List[str]=None):
        return self

    @abstractmethod
    def predict(self, X, feature_names=None, X_text: List[str]=None) -> np.ndarray[int]:
        return self

    def _set_value_acc_samples(self, X_text, y):
        """Set value and accuracy of stump.
        """
        idxs_right = self.predict(X_text).astype(bool)
        self.value = [np.mean(y[~idxs_right]), np.mean(y[idxs_right])]
        self.value_mean = np.mean(y)
        self.n_samples = [y.size - idxs_right.sum(), idxs_right.sum()]
        self.acc = accuracy_score(y, 1 * idxs_right)

class PromptStump(Stump):

    def fit(self, X, y, feature_names=None, X_text=None):
        # check input and set some attributes
        assert len(np.unique(y)) > 1, 'y should have more than 1 unique value'
        assert len(np.unique(y)) <= 2, 'only binary classification is supported'
        X, y, _ = imodels.util.arguments.check_fit_arguments(
            self, X, y, feature_names)
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # actually run fitting
        breakpoint()
        return self



class KeywordStump(Stump):

    def fit(self, X, y, feature_names=None, X_text=None):
        # check input and set some attributes
        assert len(np.unique(y)) > 1, 'y should have more than 1 unique value'
        assert len(np.unique(y)) <= 2, 'only binary classification is supported'
        X, y, _ = imodels.util.arguments.check_fit_arguments(
            self, X, y, feature_names)
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # fit stump
        if self.split_strategy == 'linear':
            self.stump_keywords_idxs = self._get_stump_keywords_linear(X, y)
        elif self.split_strategy == 'cart':
            self.stump_keywords_idxs = self._get_stump_keywords_cart(X, y)
        self.stump_keywords = self.feature_names[self.stump_keywords_idxs]

        # set value
        self._set_value_acc_samples(X_text, y)

        # checks
        if self.assert_checks:
            preds_text = self.predict(X_text)
            preds_tab = self._predict_tabular(X)
            assert np.all(
                preds_text == preds_tab), 'predicting with text and tabular should give same results'
            assert self.value[1] > self.value[0], 'right child should have greater val than left but value=' + \
                str(self.value)
            assert self.value[1] > self.value_mean, 'right child should have greater val than parent ' + \
                str(self.value)

        return self


    def predict(self, X_text: List[str]) -> np.ndarray[int]:
        """Returns prediction 1 for positive and 0 for negative.
        """
        keywords = self.stump_keywords
        ngrams_used_to_predict = max(
                [len(keyword.split(' ')) for keyword in keywords])

        def contains_any_of_keywords(text):
            text = text.lower()
            text = imodelsx.util.generate_ngrams_list(
                text,
                ngrams=ngrams_used_to_predict,
                tokenizer_ngrams=self.tokenizer,
                all_ngrams=True
            )
            for keyword in keywords:
                if keyword in text:
                    return 1
            return 0
        contains_keywords = 1 * \
            np.array([contains_any_of_keywords(x) for x in X_text])
        if self.pos_or_neg == 'pos':
            return contains_keywords
        else:
            return 1 - contains_keywords

    def _predict_tabular(self, X):
        X = imodels.util.arguments.check_fit_X(X)
        # predict whether input has any of the features in stump_keywords_idxs
        X_feats = X[:, self.stump_keywords_idxs]
        pred = np.any(X_feats, axis=1)
        if self.pos_or_neg == 'pos':
            return pred.astype(int)
        else:
            return 1 - pred

    def _get_stump_keywords_linear(self, X, y):
        # fit a linear model
        m = LogisticRegression().fit(X, y)
        m.fit(X, y)

        # find the largest magnitude coefs
        abs_feature_idxs = m.coef_.argsort().flatten()
        bot_feature_idxs = abs_feature_idxs[:self.max_features]
        top_feature_idxs = abs_feature_idxs[-self.max_features:][::-1]

        # return the features with the largest magnitude coefs
        if np.sum(abs(bot_feature_idxs)) > np.sum(abs(top_feature_idxs)):
            self.pos_or_neg = 'neg'
            return bot_feature_idxs
        else:
            self.pos_or_neg = 'pos'
            return top_feature_idxs

    def _get_stump_keywords_cart(self, X, y):
        '''Find the top self.max_features features selected by CART
        '''
        criterion_func = imodelsx.metrics.gini_binary
        
        # Calculate the gini impurity reduction for each (binary) feature in X
        impurity_reductions = []

        # whether the feature increases the likelihood of the positive class
        feature_positive = []
        y_mean = np.mean(y)
        n = y.size
        gini_impurity = 1 - criterion_func(y_mean)
        for i in range(X.shape[1]):
            x = X[:, i]
            idxs_r = x > 0.5
            idxs_l = x <= 0.5
            if idxs_r.sum() == 0 or idxs_l.sum() == 0:
                impurity_reductions.append(0)
                feature_positive.append(True)
            else:
                y_mean_l = np.mean(y[idxs_l])
                y_mean_r = np.mean(y[idxs_r])
                gini_impurity_l = 1 - criterion_func(y_mean_l)
                gini_impurity_r = 1 - criterion_func(y_mean_r)
                # print('l', indexes_l.sum(), 'r', indexes_r.sum(), 'n', n)
                impurity_reductions.append(
                    gini_impurity
                    - (idxs_l.sum() / n) * gini_impurity_l
                    - (idxs_r.sum() / n) * gini_impurity_r
                )
                feature_positive.append(y_mean_r > y_mean_l)

        impurity_reductions = np.array(impurity_reductions)
        feature_positive = np.arange(X.shape[1])[np.array(feature_positive)]

        # find the top self.max_features with the largest impurity reductions
        args_largest_reduction_first = np.argsort(impurity_reductions)[::-1]
        self.impurity_reductions = impurity_reductions[args_largest_reduction_first][:self.max_features]
        # print('\ttop_impurity_reductions', impurity_reductions[args_largest_reduction_first][:5],
        #   'max', max(impurity_reductions))
        # print(f'\t{X.shape=}')
        imp_pos_top = [
            k for k in args_largest_reduction_first
            if k in feature_positive
            and not k in STOPWORDS
        ][:self.max_features]
        imp_neg_top = [
            k for k in args_largest_reduction_first
            if not k in feature_positive
            and not k in STOPWORDS
        ][:self.max_features]

        # feat = DecisionTreeClassifier(max_depth=1).fit(X, y).tree_.feature[0]
        if np.sum(imp_pos_top) > np.sum(imp_neg_top):
            self.pos_or_neg = 'pos'
            return imp_pos_top
        else:
            self.pos_or_neg = 'neg'
            return imp_neg_top

    def __str__(self):
        keywords = self.stump_keywords
        keywords_str = ", ".join(keywords[:5])
        if len(keywords) > 5:
            keywords_str += f'...({len(keywords) - 5} more)'
        sign = {'pos': '+', 'neg': '--'}[self.pos_or_neg]
        return f'Stump(val={self.value_mean:0.2f} n={self.n_samples}) {sign} {keywords_str}'