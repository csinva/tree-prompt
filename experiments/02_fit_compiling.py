import argparse
from copy import deepcopy
from functools import partial
import logging
import random
from collections import defaultdict
from os.path import join, dirname
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.tree
import pickle as pkl
import imodelsx.data
import inspect
import os
import torch
import transformers
from tprompt.compiler.evaluator import PromptHooker
from transformers import AutoTokenizer, AutoModelForCausalLM
import tprompt.tree
import tprompt.data
import tprompt.prompts
import tprompt.model
import cache_save_utils
from tprompt.compiler import compiling
path_to_repo = dirname(dirname(os.path.abspath(__file__)))


def get_text_data(args):
    if args.dataset_name.startswith('knnp'):
        # format like knnprompting__imdb
        dataset_name = args.dataset_name.split('__')[1]
        X_train_text, X_test_text, y_train, y_test = tprompt.data.load_knnprompting_dataset(
            dataset_name, 10_000
        )
        # Essentially disable templating in favor of knnprompt templating
        args.template_data_demonstrations = '%s%s'
    else:
        # X_* are np arrays of text. y_* are np arrays of labels [0, 0, 1, ...]
        X_train_text, X_test_text, y_train, y_test = imodelsx.data.load_huggingface_dataset(
            dataset_name=args.dataset_name,
            # subsample_frac=args.subsample_frac,
            return_lists=True,
            binary_classification=args.binary_classification,
        )
    if args.truncate_example_length > 0:
        X_train_text = [x[:args.truncate_example_length]
                        for x in X_train_text]
        X_test_text = [x[:args.truncate_example_length]
                       for x in X_test_text]

    if args.subsample_train_size > 0:
        sss = min(args.subsample_train_size, len(X_train_text))
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_train_text), sss, replace=False)
        X_train_text = [X_train_text[i] for i in idx]
        y_train = y_train[idx]
    if args.subsample_test_size > 0:
        sss = min(args.subsample_test_size, len(X_test_text))
        rng = np.random.default_rng(args.seed + 1)
        idx = rng.choice(len(X_test_text), sss, replace=False)
        X_test_text = [X_test_text[i] for i in idx]
        y_test = y_test[idx]
    return X_train_text, X_test_text, y_train, y_test


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument('--dataset_name', type=str,
                        default='rotten_tomatoes', help='name of dataset')

    # training misc args
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--save_dir', type=str, default=join(path_to_repo, 'results', 'tmp'),
                        help='directory for saving')

    # model args
    parser.add_argument('--model_name', type=str, default='single_prompt',
                        choices=['single_prompt'],
                        help='name of model. "Manual" specifies that it first calculates all features and then uses sklearn tree')
    parser.add_argument('--prompt_at_start_or_end', type=str, default='end',
                        choices=['start', 'end'], help='where to put prompt')
    parser.add_argument('--checkpoint', type=str, default='gpt2',
                        help='the underlying model used for prediction (or for constructing features from prompt)')
    parser.add_argument('--verbalizer_num', type=int, default=0,
                        help='which verbalizer to use')
    parser.add_argument('--prompt_source', type=str, default='manual', choices=['manual', 'data_demonstrations'],
                        help='''where prompts come from. Setting to manual would use PROMPTS_MOVIE_0, and data_demonstrations
                        would use example demonstrations from training set.''')
    parser.add_argument('--template_data_demonstrations', type=str,
                        default='Input: %s\nOutput:%s', help='template, only for --prompt_source data_demonstrations!')
    parser.add_argument('--num_data_demonstrations_per_class', type=int, default=1,
                        help='If prompt source is data_demonstrations, how many to include per class')
    parser.add_argument('--truncate_example_length', type=int, default=3000,
                        help='Max length of characters for each input')
    parser.add_argument('--binary_classification', type=int, default=1,
                        help='Whether to truncate dataset to binary classification')
    parser.add_argument('--subsample_train_size', type=int,
                        default=300, help='Amount to subsample the training data')
    parser.add_argument('--subsample_test_size', type=int,
                        default=300, help='Amount to subsample the testing data')

    parser.add_argument('--num_prompts', type=int,
                        default=40, help='number of prompts to use')
    parser.add_argument('--filter_by_median', type=int, default=10,
                        help='If > 0, number of prompts to keep (closest to the median)')
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
    if args.binary_classification == 0 and args.dataset_name in ['sst2', 'rotten_tomatoes', 'imdb']:
        logging.info(
            f'Skipping {args.dataset_name} since binary_classification=0')
        exit(0)
    if args.dataset_name == 'emotion' and args.verbalizer_num == 1:
        logging.info(f'Skipping {args.dataset_name} since verbalizer_num=1')
        exit(0)
    args.verbalizer = tprompt.prompts.get_verbalizer(args)

    # newly added this for compatibility!
    if not isinstance(args.verbalizer, dict):
        args.verbalizer = {k: args.verbalizer.verbalizer[args.verbalizer.id2label[k]]
                           for k in args.verbalizer.id2label.keys()}

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir)
    os.makedirs(save_dir_unique, exist_ok=True)

    if args.use_cache and already_cached:
        logging.info(
            f'cached version exists! Successfully skipping :)\n\n\n')
        exit(0)
    for k in sorted(vars(args)):
        logger.info('\t' + k + ' ' + str(vars(args)[k]))
    logging.info(f'\n\n\tsaving to ' + save_dir_unique + '\n')

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    transformers.set_seed(args.seed)
    r = defaultdict(list)

    # load and process text data (truncate, subsample)
    X_train_text, X_test_text, y_train, y_test = get_text_data(args)

    # get prompts
    prompts = tprompt.prompts.get_prompts(
        args, X_train_text, y_train, args.verbalizer, seed=1  # note, not passing seed here!
    )[:args.num_prompts]
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    lengths_in_tokens = [len(tok.encode(x)) for x in prompts]

    # filter by median
    if args.filter_by_median > 0:
        median = np.median(lengths_in_tokens)
        idx = np.argsort(np.abs(np.array(lengths_in_tokens) - median))
        prompts = [prompts[i] for i in idx[:args.filter_by_median]]
        lengths_in_tokens = [lengths_in_tokens[i]
                             for i in idx[:args.filter_by_median]]
    print('lengths_in_tokens', lengths_in_tokens)

    # compile prompts
    # print('prompts', len(prompts), lens)
    avg_soft_prompt = compiling.get_avg_soft_prompt(args.checkpoint, prompts)
    # print('avg_soft_prompt', avg_soft_prompt.shape)

    # score avg model
    longest_prompt_idx = np.argmax(lengths_in_tokens)
    kwargs = {
        'checkpoint': args.checkpoint,
        'verbalizer': args.verbalizer,
        'random_state': args.seed,
        'prompt_at_start_or_end': args.prompt_at_start_or_end,
        'prompt_template': "{example}{prompt}",
    }
    m = PromptHooker(
        prompts=[prompts[longest_prompt_idx]],
        hook_weights=avg_soft_prompt,
        verbose=False,
        cache_prompt_features_dir=join(args.cache_prompt_features_dir, 'avg'),
        **kwargs,
    )
    m.fit(X_train_text, y_train)
    r['acc_compiled'] = m.prompt_accs_[0]

    # score individual models
    m = PromptHooker(
        prompts=prompts,
        hook_weights=None,
        verbose=False,
        cache_prompt_features_dir=args.cache_prompt_features_dir,
        **kwargs,
    )
    m.fit(X_train_text, y_train)
    accs_single = deepcopy(m.prompt_accs_)

    # set up saving dictionary + save params file
    r.update(vars(args))
    r['accs_single'] = accs_single
    r['acc_single_mean'] = np.mean(accs_single)
    r['acc_single_max'] = np.max(accs_single)
    # print('Avg', r['acc_compiled'], 'Sing', r['accs_single'])

    r['prompts'] = prompts
    r['save_dir_unique'] = save_dir_unique
    pkl.dump(r, open(join(save_dir_unique, 'results.pkl'), 'wb'))
    # pkl.dump(model, open(join(save_dir_unique, 'model.pkl'), 'wb'))
    logging.info('Succesfully completed :)\n\n')
