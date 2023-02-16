import numpy as np
import tprompt.stump
import tprompt.tree
import tprompt.data
import random
from tqdm import tqdm
import imodelsx.data
import sklearn.tree
import tprompt.prompts
from transformers import AutoModelForCausalLM
import logging

def seed_and_get_tiny_data(seed=1, subsample_frac=0.05):
    np.random.seed(seed)
    random.seed(seed)
    X_train_text, X_test_text, y_train, y_test = imodelsx.data.load_huggingface_dataset(
        dataset_name='rotten_tomatoes', subsample_frac=subsample_frac, return_lists=True)
    X_train, _, feature_names = \
        tprompt.data.convert_text_data_to_counts_array(
            X_train_text, X_test_text, ngrams=1)
    return X_train_text, X_test_text, y_train, X_train, y_test, feature_names


def test_stump_improves_acc(split_strategy='iprompt'):
    X_train_text, X_test_text, y_train, X_train, y_test, feature_names = seed_and_get_tiny_data(
        seed=1, subsample_frac=0.05)
    stump_cls = tprompt.stump.PromptStump
    class args:
        dataset_name = 'rotten_tomatoes'
        verbose = True
    m = stump_cls(
        args=args(),
        split_strategy=split_strategy,
        checkpoint='EleutherAI/gpt-j-6B',
        # checkpoint='gpt2',
        assert_checks=True,
    )

    # test actually fitting
    m.fit(
            X_text=X_train_text,
            y=y_train,
            feature_names=feature_names,
            X=X_train,
        )
    print('top prompt', m.prompt)
    print('found prompts', m.prompts)
    preds = m.predict(X_train_text)
    acc_baseline = max(y_train.mean(), 1 - y_train.mean())
    acc = np.mean(preds == y_train)
    assert acc > acc_baseline, f'stump must improve train acc but {acc:0.2f} <= {acc_baseline:0.2f}'
    print(acc, acc_baseline)


def test_stump_manual():
    X_train_text, X_test_text, y_train, X_train, y_test, feature_names = seed_and_get_tiny_data(
        seed=1, subsample_frac=0.05)
    X_cv_text = X_test_text
    y_cv = y_test
    
    class args:
        prompt = 'Placeholder' # we set this in the loop below
        verbalizer = {
            0: ' Negative.',
            1: ' Positive.',
        }
        batch_size = 32
        
    m = tprompt.stump.PromptStump(
        args=args(),
        split_strategy='manual', # 'manual' specifies that we use args.prompt
        checkpoint='gpt2', # EleutherAI/gpt-j-6B
        assert_checks=True,
    )

    # test different manual stumps
    prompts = tprompt.prompts.PROMPTS_MOVIE_0[:2]

    prompt_features_train = np.zeros((len(X_train_text), len(prompts)))
    prompt_features_test = np.zeros((len(X_test_text), len(prompts)))
    prompt_features_cv = np.zeros((len(X_cv_text), len(prompts)))
    accs_cv = np.zeros(len(prompts))
    for i, p in enumerate(tqdm(prompts)):
        m.prompt = p
        
        acc_baseline = max(y_train.mean(), 1 - y_train.mean())
        logging.info('prompt ' + p)
        preds_train = m.predict(X_train_text)
        acc_train = np.mean(preds_train == y_train)
        prompt_features_train[:, i] = preds_train
        logging.info(f'\tacc_train {acc_train:0.3f} baseline: {acc_baseline:0.3f}')

        preds_test = m.predict(X_test_text)
        acc_test = np.mean(preds_test == y_test)
        prompt_features_test[:, i] = preds_test
        logging.info(f'\tacc_test {acc_test:0.3f}')

        preds_cv = m.predict(X_cv_text)
        acc_cv = np.mean(preds_cv == y_cv)
        prompt_features_cv[:, i] = preds_cv
        logging.info(f'\tacc_cv {acc_cv:0.3f}')
        accs_cv[i] = acc_cv

    a = np.argsort(accs_cv.flatten())[::-1]
    prompt_features_train[:, a], prompt_features_test[:, a], prompt_features_cv[:, a], np.array(prompts)[:, a].tolist()

# def test_tree_monotonic_in_depth(split_strategy='linear'):
#     X_train_text, X_test_text, y_train, X_train, y_test, feature_names = seed_and_get_tiny_data()

#     accs = [y_train.mean()]
#     # for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
#     for max_depth in [1, 2, 3]:
#         class args:
#             dataset_name = 'rotten_tomatoes'
#         m = tprompt.tree.Tree(
#             args=args(),
#             max_depth=max_depth,
#             split_strategy=split_strategy,
#             verbose=True,
#             assert_checks=True,
#         )
#         m.fit(X_text=X_train_text, X=X_train, y=y_train, feature_names=feature_names)
#         preds = m.predict(X_text=X_train_text)
#         accs.append(np.mean(preds == y_train))
#         assert accs[-1] >= accs[-2], 'train_acc must monotonically increase with max_depth ' + \
#             str(accs)
#         print(m)
#         print('\n')

if __name__ == '__main__':
    split_strategy = 'iprompt'
    # test_stump_improves_acc(split_strategy)
    # test_tree_monotonic_in_depth(split_strategy)
    test_stump_manual()
    # test_tree_manual()
