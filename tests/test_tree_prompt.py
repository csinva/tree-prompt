import numpy as np
import tprompt.stump
import tprompt.tree
import tprompt.data
import random
import imodelsx.data


def seed_and_get_tiny_data(seed=1, subsample_frac=0.05):
    np.random.seed(seed)
    random.seed(seed)
    X_train_text, X_test_text, y_train, y_test = imodelsx.data.load_huggingface_dataset(
        dataset_name='rotten_tomatoes', subsample_frac=subsample_frac, return_lists=True)
    X_train, _, feature_names = \
        tprompt.data.convert_text_data_to_counts_array(
            X_train_text, X_test_text, ngrams=1)
    return X_train_text, X_test_text, y_train, X_train, y_test, feature_names


def test_stump_always_improves_acc(split_strategy='iprompt'):
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


    # test just predicting with different prompts
    for prompt in [
        'What is the sentiment expressed by the reviewer for the movie?',
    ]:
        m.prompt = prompt
        preds = m.predict(X_train_text)
        acc_baseline = max(y_train.mean(), 1 - y_train.mean())
        acc = np.mean(preds == y_train)
        assert acc > acc_baseline, f'stump must acc but {acc:0.2f} <= {acc_baseline:0.2f}'
        print(acc, acc_baseline)

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
    test_stump_always_improves_acc(split_strategy)
    # test_tree_monotonic_in_depth(split_strategy)
