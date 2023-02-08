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
import pickle as pkl
import imodelsx.data
import inspect
import os

import tprompt.tree
import tprompt.data
import cache_save_utils
path_to_repo = dirname(dirname(os.path.abspath(__file__)))


def fit_model(model, X_train, X_train_text, y_train, feature_names, r):
    # fit the model
    fit_parameters = inspect.signature(model.fit).parameters.keys()
    kwargs = {}
    if 'feature_names' in fit_parameters and feature_names is not None:
        kwargs['feature_names'] = feature_names
    if 'X_text' in fit_parameters:
        kwargs['X_text'] = X_train_text
    model.fit(X=X_train, y=y_train, **kwargs)

    if hasattr(model, 'prompts_list'):
        r['prompts_list'] = model.prompts_list

    return r, model

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
    parser.add_argument('--subsample_frac', type=float,
                        default=1, help='fraction of samples to use')

    # training misc args
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--save_dir', type=str, default=join(path_to_repo, 'results', 'tmp'),
                        help='directory for saving')

    # model args
    parser.add_argument('--split_strategy', type=str, choices=['iprompt', 'cart', 'linear'],
                        default='iprompt', help='strategy to use to split each stump')
    parser.add_argument('--max_depth', type=int,
                        default=2, help='max depth of tree')
    parser.add_argument('--checkpoint', type=str, default='EleutherAI/gpt-j-6B',
                        help='the underlying model used for prediction')
    parser.add_argument('--checkpoint_prompting', type=str, default='EleutherAI/gpt-j-6B',
                        help='the model used for finding the prompt')
            
    return parser

def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)
    """
    parser.add_argument('--use_cache', type=int, default=1, choices=[0, 1],
                        help='whether to check for cache')
    parser.add_argument('--use_verbose', type=int, default=1, choices=[0, 1],
                        help='whether to print verbosely')
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
        dataset_name=args.dataset_name, subsample_frac=args.subsample_frac,
        return_lists=True, binary_classification=True,
    )
    # get converted tabular data too just in case
    X_train, X_test, feature_names = \
        tprompt.data.convert_text_data_to_counts_array(
            X_train_text, X_test_text, ngrams=2)
    X_train, X_cv, X_train_text, X_cv_text, y_train, y_cv = train_test_split(
        X_train, X_train_text, y_train, test_size=0.33, random_state=args.seed)

    # load model
    model = tprompt.tree.Tree(
        args=args,
        max_depth=args.max_depth,
        split_strategy=args.split_strategy,
        verbose=args.use_verbose,
        checkpoint=args.checkpoint,
        checkpoint_prompting=args.checkpoint_prompting,
    )

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r['save_dir_unique'] = save_dir_unique
    cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname='params.json', r=r)

    # fit
    r, model = fit_model(model, X_train, X_train_text, y_train, feature_names, r)
    
    # evaluate
    r = evaluate_model(
        model,
        X_train, X_cv, X_test,
        X_train_text, X_cv_text, X_test_text,
        y_train, y_cv, y_test, r
    )

    # save results
    pkl.dump(r, open(join(save_dir_unique, 'results.pkl'), 'wb'))
    pkl.dump(model, open(join(save_dir_unique, 'model.pkl'), 'wb'))
    logging.info('Succesfully completed :)\n\n')
