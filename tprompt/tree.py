from typing import List
import numpy as np
import imodels
import imodelsx.util
from tprompt.stump import KeywordStump, PromptStump, Stump
import tprompt.data
import logging
import warnings

class Tree:
    def __init__(
        self,
        max_depth: int = 3,
        split_strategy: str='iprompt',
        verbose=True,
        tokenizer=None,
        assert_checks=True,
    ):
        '''
        Params
        ------
        max_depth: int
            Maximum depth of the tree.
        split_strategy: str
            'iprompt' - use prompted language model to split
            'cart' - use cart to split
            'linear' - use linear to split
        verbose: bool
        tokenizer
        assert_checks: bool
            Whether to run checks during fitting
        '''
        self.max_depth = max_depth
        self.split_strategy = split_strategy
        self.verbose = verbose
        self.assert_checks  = assert_checks
        if tokenizer is None:
            self.tokenizer = imodelsx.util.get_spacy_tokenizer(convert_output=False)
        else:
            self.tokenizer = tokenizer

    def fit(self, X=None, y=None, feature_names=None, X_text: List[str]=None):
        if X is None and X_text:
            warnings.warn("X is not passed, defaulting to generating unigrams from X_text")
            X, _, feature_names = tprompt.data.convert_text_data_to_counts_array(X_text, [], ngrams=1)

        # check and set some attributes
        X, y, _ = imodels.util.arguments.check_fit_arguments(
            self, X, y, feature_names)
        if isinstance(X_text, list):
            X_text = np.array(X_text).flatten()
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # set up arguments
        stump_kwargs = dict(
            tokenizer=self.tokenizer,
            split_strategy=self.split_strategy,
            assert_checks=self.assert_checks,
            verbose=self.verbose,
        )
        if self.split_strategy in ['iprompt']:
            stump_class = PromptStump
        else:
            stump_class = KeywordStump


        # fit root stump
        stump = stump_class(**stump_kwargs).fit(
            X, y,
            feature_names=self.feature_names,
            X_text=X_text
        )
        stump.idxs = np.ones(X.shape[0], dtype=bool)
        self.root_ = stump

        # recursively fit stumps and store as a decision tree
        stumps_queue = [stump]
        i = 0
        depth = 1
        while depth < self.max_depth:
            stumps_queue_new = []
            for stump in stumps_queue:
                stump = stump
                if self.verbose:
                    logging.debug(f'Splitting on {depth=} stump_num={i} {stump.idxs.sum()=}')
                idxs_pred = stump.predict(X_text=X_text) > 0.5
                for idxs_p, attr in zip([~idxs_pred, idxs_pred], ['child_left', 'child_right']):
                    # for idxs_p, attr in zip([idxs_pred], ['child_right']):
                    idxs_child = stump.idxs & idxs_p
                    if self.verbose:
                        logging.debug(f'\t{idxs_pred.sum()=} {idxs_child.sum()=}', len(np.unique(y[idxs_child])))
                    if idxs_child.sum() > 0 \
                        and idxs_child.sum() < stump.idxs.sum() \
                            and len(np.unique(y[idxs_child])) > 1:

                        # sometimes this fails to find a split that partitions any points at all
                        stump_child = stump_class(**stump_kwargs).fit(
                            X[idxs_child], y[idxs_child],
                            X_text=X_text[idxs_child],
                            feature_names=self.feature_names,
                        )

                        # set the child stump
                        stump_child.idxs = idxs_child
                        acc_tree_baseline = np.mean(self.predict(
                            X_text[idxs_child]) == y[idxs_child])
                        if attr == 'child_left':
                            stump.child_left = stump_child
                        else:
                            stump.child_right = stump_child
                        stumps_queue_new.append(stump_child)
                        if self.verbose:
                            logging.debug(f'\t\t {stump.stump_keywords} {stump.pos_or_neg}')
                        i += 1

                        ######################### checks ###########################
                        # if self.refinement_strategy == 'None' and self.assert_checks:
                        if self.assert_checks:
                            # check acc for the points in this stump
                            acc_tree = np.mean(self.predict(
                                X_text[idxs_child]) == y[idxs_child])
                            assert acc_tree >= acc_tree_baseline, f'stump acc {acc_tree:0.3f} should be > after adding child {acc_tree_baseline:0.3f}'

                            # check total acc
                            acc_total_baseline = max(y.mean(), 1 - y.mean())
                            acc_total = np.mean(self.predict(X_text) == y)
                            assert acc_total >= acc_total_baseline, f'total acc {acc_total:0.3f} should be > after adding child {acc_total_baseline:0.3f}'

                            # check that stumptrain acc improved over this set
                            # not necessarily going to improve total acc, since the stump always predicts 0/1
                            # even though the correct answer might be always 0 or always be 1
                            acc_child_baseline = min(
                                y[idxs_child].mean(), 1 - y[idxs_child].mean())
                            assert stump_child.acc > acc_child_baseline, f'acc {stump_child.acc:0.3f} should be > baseline {acc_child_baseline:0.3f}'


            stumps_queue = stumps_queue_new
            depth += 1

        return self

    def predict_proba(self, X_text: List[str] = None):
        preds = []
        for x_t in X_text:

            # prediction for single point
            stump = self.root_
            while stump:
                # 0 or 1 class prediction here
                pred = stump.predict(X_text=[x_t])[0]
                value = stump.value

                if pred > 0.5:
                    stump = stump.child_right
                    value = value[1]
                else:
                    stump = stump.child_left
                    value = value[0]

                if stump is None:
                    preds.append(value)
        preds = np.array(preds)
        probs = np.vstack((1 - preds, preds)).transpose()  # probs (n, 2)
        return probs

    def predict(self, X_text: List[str] = None) -> np.ndarray[int]:
        preds_bool = self.predict_proba(X_text)[:, 1]
        return (preds_bool > 0.5).astype(int)
    

    def __str__(self):
        s = f'> Tree(max_depth={self.max_depth} split_strategy={self.split_strategy})\n> ------------------------------------------------------\n'
        return s + self.viz_tree()

    def viz_tree(self, stump: Stump=None, depth: int=0, s: str='') -> str:
        if stump is None:
            stump = self.root_
        s += '   ' * depth + str(stump) + '\n'
        if stump.child_left:
            s += self.viz_tree(stump.child_left, depth + 1)
        else:
            s += '   ' * (depth + 1) + f'Neg n={stump.n_samples[0]} val={stump.value[0]:0.3f}' + '\n'
        if stump.child_right:
            s += self.viz_tree(stump.child_right, depth + 1)
        else:
            s += '   ' * (depth + 1) + f'Pos n={stump.n_samples[1]} val={stump.value[1]:0.3f}' + '\n'
        return s
