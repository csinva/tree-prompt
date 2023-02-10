from typing import List
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import imodels
import imodelsx.util
import imodelsx.metrics
import torch.cuda
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from abc import ABC, abstractmethod


class Stump(ABC):
    def __init__(
        self,
        args,
        split_strategy: str='iprompt',
        tokenizer=None,
        max_features=10,
        assert_checks: bool=False,
        verbose: bool=True,
        checkpoint: str='EleutherAI/gpt-j-6B',
        checkpoint_prompting: str='EleutherAI/gpt-j-6B',
    ):
        """Fit a single stump.
        Can use tabular features...
            Currently only supports binary classification with binary features.
        Params
        ------
        split_strategy: str
            'iprompt' - use iprompt to split
            'manual' - use passed prompt in args.prompt
            'cart' - use cart to split
            'linear' - use linear to split
        max_features: int
            used by StumpTabular to decide how many features to save
        checkpoint: str
            the underlying model used for prediction
        checkpoint_prompting: str
            the model used for finding the prompt
        """
        self.args = args
        assert split_strategy in ['iprompt', 'cart', 'linear', 'manual']
        self.split_strategy = split_strategy
        self.assert_checks = assert_checks
        self.verbose = verbose
        self.max_features = max_features 
        self.checkpoint = checkpoint
        self.checkpoint_prompting = checkpoint_prompting
        if tokenizer is None:
            self.tokenizer = imodelsx.util.get_spacy_tokenizer(convert_output=False)
        else:
            self.tokenizer = tokenizer

        # tree stuff
        self.child_left = None
        self.child_right = None
    
    @abstractmethod
    def fit(self, X_text: List[str], y: List[int], feature_names=None, X=None):
        return self

    @abstractmethod
    def predict(self, X_text: List[str]) -> np.ndarray[int]:
        return self

    def _set_value_acc_samples(self, X_text, y):
        """Set value and accuracy of stump.
        """
        idxs_right = self.predict(X_text).astype(bool)
        self.value = [np.mean(y[~idxs_right]), np.mean(y[idxs_right])]
        self.value_mean = np.mean(y)
        self.n_samples = [y.size - idxs_right.sum(), idxs_right.sum()]
        self.acc = accuracy_score(y, 1 * idxs_right)


DATA_OUTPUT_STRINGS = {
    'rotten_tomatoes': {
        0: ' Negative.',
        1: ' Positive.',
    }
}
class PromptStump(Stump):

    def __init__(self, *args, **kwargs):
        super(PromptStump, self).__init__(*args, **kwargs)
        if self.verbose:
            logging.info(f'Loading model {self.checkpoint}')
            # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = 'cpu' # some error when this is specified to cuda, idk what it is...
            self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=False)

    def fit(self, X_text: List[str], y, feature_names=None, X=None):
        # check input and set some attributes
        assert len(np.unique(y)) > 1, 'y should have more than 1 unique value'
        X, y, _ = imodels.util.arguments.check_fit_arguments(
            self, X, y, feature_names)
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # actually run fitting
        input_strings = X_text
        verbalizer_dict = self._get_verbalizer()
        output_strings = [verbalizer_dict[int(yi)] for yi in y]

        # get prompt
        if self.split_strategy == 'manual':
            self.prompt = self.args.prompt
        else:
            self.model = self.model.to('cpu')
            prompts, metadata = imodelsx.explain_dataset_iprompt(
                input_strings=input_strings,
                output_strings=output_strings,
                checkpoint=self.checkpoint, # which language model to use
                num_learned_tokens=6, # how long of a prompt to learn
                n_shots=5, # number of examples in context
                n_epochs=5, # how many epochs to search
                batch_size=16, # batch size for iprompt
                llm_float16=True, # whether to load the model in float_16
                verbose=0, # how much to print
                prefix_before_input=False,
                # max_n_datapoints=100, # restrict this for now
            )
            print('prompts', prompts)
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

            # save stuff
            self.prompt = prompts[0]
            self.prompts = prompts
            self.meta = metadata

        # set value (calls self.predict)
        self._set_value_acc_samples(X_text, y)
        
        return self

    def predict(self, X_text: List[str]) -> np.ndarray[int]:
        '''todo: pass in model here so we can share it across all stumps
        '''
        self.model = self.model.to('cuda')
        preds_proba = self.predict_proba(X_text)
        return np.argmax(preds_proba, axis=1)

    def predict_proba(self, X_text: List[str]) -> np.ndarray[float]:
        '''todo: pass in model here so we can share it across all stumps
        '''
        target_strs = list(self._get_verbalizer().values())
        
        # only predict based on first token of output string
        target_token_ids = list(map(self._get_first_token_id, target_strs))
        preds = np.zeros((len(X_text), len(target_token_ids)))
        for i, x in enumerate(X_text):
            preds[i] = self._get_logit_for_target_tokens(x, target_token_ids)

        # return the class with the highest logit
        return softmax(preds, axis=1)

    def _get_logit_for_target_tokens(self, prompt: str, target_token_ids: List[int]) -> np.ndarray[float]:
        """Get logits for each target token
        This can fail when token_output_ids represents multiple tokens
        So things get mapped to the same id representing "unknown"
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs)['logits'].detach()  # shape is (batch_size, seq_len, vocab_size)
        # token_output_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        logit_targets = [logits[0, -1, token_output_id].item() for token_output_id in target_token_ids]
        return np.array(logit_targets)

    def _get_first_token_id(self, prompt: str) -> str:
        """Get first token id in prompt
        """
        return self.tokenizer(prompt)['input_ids'][0]

    def _get_verbalizer(self):
        if hasattr(self.args, 'verbalizer') and self.args.verbalizer is not None:
            return self.args.verbalizer
        else:
            return DATA_OUTPUT_STRINGS[self.args.dataset_name]
    
    def __str__(self):
        return f'PromptStump(val={self.value_mean:0.2f} prompt={self.prompt})'


class KeywordStump(Stump):

    def fit(self, X_text: List[str], y, feature_names=None, X=None):
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
            and not k in imodelsx.util.STOPWORDS
        ][:self.max_features]
        imp_neg_top = [
            k for k in args_largest_reduction_first
            if not k in feature_positive
            and not k in imodelsx.util.STOPWORDS
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
