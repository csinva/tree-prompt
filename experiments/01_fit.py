import argparse
from copy import deepcopy
from functools import partial
import logging
import random
from collections import defaultdict
from os.path import join, dirname
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
import sklearn.ensemble
import sklearn.tree
import pickle as pkl
import imodelsx.data
import inspect
import os

import tprompt.tree
import tprompt.data
import tprompt.prompts
import tprompt.ensemble
import cache_save_utils
path_to_repo = dirname(dirname(os.path.abspath(__file__)))


def get_verbalizer(args):
    VERB0 = {0: ' Negative.', 1: ' Positive.'}
    VERB1 = {0: ' No.', 1: ' Yes.', }
    VERB_LIST_DEFAULT = [VERB0, VERB1]
    DATA_OUTPUT_STRINGS = {
        'rotten_tomatoes': [VERB0, VERB1],
        'sst2': [VERB0, VERB1],
        'emotion': [VERB0, VERB1],
        'financial_phrasebank': [VERB0, VERB1],
    }
    return DATA_OUTPUT_STRINGS.get(args.dataset_name, VERB_LIST_DEFAULT)[args.verbalizer_num]


def evaluate_model(model, X_train, X_cv, X_test,
                   X_train_text, X_cv_text, X_test_text,
                   y_train, y_cv, y_test, r):
    """Evaluate model performance on each split
    """
    metrics = {
        'accuracy': accuracy_score,
        'precision': partial(precision_score, zero_division=0),
        'recall': partial(recall_score, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score,
    }
    metrics_proba = {
        'roc_auc': roc_auc_score,
        'brier_score_loss': brier_score_loss,
    }
    for split_name, (X_text_, X_, y_) in zip(['train', 'cv', 'test'],
                                             [(X_train_text, X_train, y_train),
                                             (X_cv_text, X_cv, y_cv),
                                             (X_test_text, X_test, y_test)]):
        # metrics discrete
        predict_parameters = inspect.signature(model.predict).parameters.keys()
        if 'X_text' in predict_parameters:
            y_pred_ = model.predict(X_text=X_text_).astype(int)
        else:
            y_pred_ = model.predict(X_)
        for metric_name, metric_fn in metrics.items():
            r[f'{metric_name}_{split_name}'] = metric_fn(y_, y_pred_)

        # metrics proba
        if hasattr(model, 'predict_proba'):
            if 'X_text' in predict_parameters:
                y_pred_proba_ = model.predict_proba(X_text=X_text_)[:, 1]
            else:
                y_pred_proba_ = model.predict_proba(X_)[:, 1]
        for metric_name, metric_fn in metrics_proba.items():
            r[f'{metric_name}_{split_name}'] = metric_fn(y_, y_pred_proba_)

    return r

# initialize args


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument('--dataset_name', type=str,
                        default='rotten_tomatoes', help='name of dataset')
    # parser.add_argument('--subsample_frac', type=float,
                        # default=1, help='fraction of samples to use')

    # training misc args
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--save_dir', type=str, default=join(path_to_repo, 'results', 'tmp'),
                        help='directory for saving')

    # model args
    parser.add_argument('--model_name', type=str, default='tprompt',
                        choices=[
                            'tprompt',
                            'manual_tree', 'manual_ensemble', 'manual_boosting',
                            'manual_gbdt' # manual_gbdt will ignore other params like num_prompts
                        ],
                        help='name of model. "Manual" specifies that it first calculates all features and then uses sklearn tree')
    parser.add_argument('--split_strategy', type=str, choices=['iprompt', 'cart', 'linear'],
                        default='iprompt', help='strategy to use to split each stump')
    parser.add_argument('--max_depth', type=int,
                        default=2, help='max depth of tree')
    parser.add_argument('--num_prompts', type=int,
                        default=1, help='only for manual things or data demonstrations (not applied when model_name==tprompt)')
    parser.add_argument('--checkpoint', type=str, default='EleutherAI/gpt-j-6B',
                        help='the underlying model used for prediction (or for constructing features from prompt)')
    parser.add_argument('--checkpoint_prompting', type=str, default='EleutherAI/gpt-j-6B',
                        help='the model used for finding the prompt')
    parser.add_argument('--verbalizer_num', type=int, default=0,
                        help='which verbalizer to use')
    parser.add_argument('--prompt_source', type=str, default='manual', choices=['manual', 'data_demonstrations'],
                        help='''where prompts come from. Setting to manual would use PROMPTS_MOVIE_0, and data_demonstrations
                        would use example demonstrations from training set.''')
    parser.add_argument('--template_data_demonstrations', type=str,
                        default='Input: %s\nOutput:%s', help='template, only for --prompt_source data_demonstrations!')

    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldn't use when checking cache)
    """
    parser.add_argument('--use_cache', type=int, default=1, choices=[0, 1],
                        help='whether to check for cache')
    parser.add_argument('--use_verbose', type=int, default=1, choices=[0, 1],
                        help='whether to print verbosely')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for manual_tree feature extraction')
    parser.add_argument('--cache_prompt_features_dir', type=str, default='/home/chansingh/mntv1/tree-prompt/cache_prompt_features',
                        help='which directory to cache prompt features into')
    return parser


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir)

    if args.use_cache and already_cached:
        logging.info(
            f'cached version exists! Successfully skipping :)\n\n\n')
        exit(0)
    for k in sorted(vars(args)):
        logger.info('\t' + k + ' ' + str(vars(args)[k]))
    logging.info(f'\n\n\tsaving to ' + save_dir_unique + '\n')

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # load text data
    X_train_text, X_test_text, y_train, y_test = imodelsx.data.load_huggingface_dataset(
        dataset_name=args.dataset_name,
        # subsample_frac=args.subsample_frac,
        return_lists=True,
        binary_classification=True,
    )

    # get converted tabular data
    X_train, X_test, feature_names = \
        tprompt.data.convert_text_data_to_counts_array(
            X_train_text, X_test_text, ngrams=2)
    if args.model_name.startswith('manual'):
        X_train, X_test, feature_names = \
            tprompt.prompts.engineer_prompt_features(
                args, X_train_text, X_test_text, y_train, y_test)
        
    # split (could subsample here too)
    X_train, X_cv, X_train_text, X_cv_text, y_train, y_cv = train_test_split(
        X_train, X_train_text, y_train, test_size=0.33, random_state=args.seed)
    args.verbalizer = get_verbalizer(args)

    # load model
    if args.model_name == 'tprompt':
        model = tprompt.tree.Tree(
            args=args,
            max_depth=args.max_depth,
            split_strategy=args.split_strategy,
            verbose=args.use_verbose,
            checkpoint=args.checkpoint,
            checkpoint_prompting=args.checkpoint_prompting,
        )
    elif args.model_name == 'manual_tree':
        model = sklearn.tree.DecisionTreeClassifier(
            max_leaf_nodes=args.num_prompts + 1,
            random_state=args.seed,
        )
    elif args.model_name == 'manual_ensemble':
        model = tprompt.ensemble.IdentityEnsembleClassifier(
            n_estimators=args.num_prompts,
        )
    elif args.model_name == 'manual_boosting':
        model = tprompt.ensemble.IdentityEnsembleClassifier(
            n_estimators=args.num_prompts,
            boosting=True,
        )
    elif args.model_name == 'manual_gbdt':
        model = sklearn.ensemble.GradientBoostingClassifier(
            random_state=args.seed,
        )

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r['save_dir_unique'] = save_dir_unique
    cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname='params.json', r=r)

    # fit the model
    fit_parameters = inspect.signature(model.fit).parameters.keys()
    kwargs = {}
    if 'feature_names' in fit_parameters and feature_names is not None:
        kwargs['feature_names'] = feature_names
    if 'X_text' in fit_parameters:
        kwargs['X_text'] = X_train_text
    model.fit(X=X_train, y=y_train, **kwargs)

    # evaluate
    r = evaluate_model(
        model,
        X_train, X_cv, X_test,
        X_train_text, X_cv_text, X_test_text,
        y_train, y_cv, y_test, r
    )

    # save results
    if hasattr(model, 'prompts_list'):
        r['prompts_list'] = model.prompts_list
        r['prompt'] = r['prompts_list'][0]
    # r['feature_names'] = feature_names
    if isinstance(model, sklearn.tree.DecisionTreeClassifier):
        r['str_tree'] = sklearn.tree.export_text(
            model, feature_names=feature_names)
    else:
        r['str_tree'] = str(model)

    r['feature_names'] = feature_names
    pkl.dump(r, open(join(save_dir_unique, 'results.pkl'), 'wb'))
    pkl.dump(model, open(join(save_dir_unique, 'model.pkl'), 'wb'))
    logging.info('Succesfully completed :)\n\n')
