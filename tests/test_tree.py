import numpy as np
import tprompt.stump
import tprompt.tree
import tprompt.data
import random
import imodelsx.data


def seed_and_get_tiny_data(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    X_train_text, X_test_text, y_train, y_test = imodelsx.data.load_huggingface_dataset(
        dataset_name='rotten_tomatoes', subsample_frac=0.05, return_lists=True)
    X_train, _, feature_names = \
        tprompt.data.convert_text_data_to_counts_array(
            X_train_text, X_test_text, ngrams=1)
    return X_train_text, X_test_text, y_train, X_train, y_test, feature_names


def test_stump_always_improves_acc():
    for seed in range(2):
        X_train_text, X_test_text, y_train, X_train, y_test, feature_names = seed_and_get_tiny_data(seed=seed)
        m = tprompt.stump.StumpTabular(
            split_strategy='linear',
            assert_checks=True,
        ).fit(
            X_train, y_train, feature_names, X_train_text)
        preds = m.predict(X_text=X_train_text)
        acc_baseline = max(y_train.mean(), 1 - y_train.mean())
        acc = np.mean(preds == y_train)
        assert acc > acc_baseline, 'stump must improve train acc'
        print(acc, acc_baseline)

"""
def test_tree_monotonic_in_depth(refinement_strategy='None', max_features=1, embs_manager=None):
    X_train_text, X_test_text, y_train, X_train, y_test, feature_names = seed_and_get_tiny_data()

    accs = [y_train.mean()]
    # for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    for max_depth in [1, 2, 3]:
        m = llm_tree.tree.Tree(
            max_depth=max_depth,
            split_strategy='cart',
            max_features=max_features,
            verbose=False,
            refinement_strategy=refinement_strategy,
            assert_checks=True,
            embs_manager=embs_manager,
        )
        m.fit(X_train, y_train, feature_names, X_train_text)
        preds = m.predict(X_text=X_train_text)
        accs.append(np.mean(preds == y_train))
        assert accs[-1] >= accs[-2], 'train_acc must monotonically increase with max_depth ' + \
            str(accs)
        print(m)
        print('\n')
"""

if __name__ == '__main__':
    test_stump_always_improves_acc()


    # for refinement_strategy in ['None', 'llm']:
    #     test_tree_monotonic_in_depth(
    #         refinement_strategy=refinement_strategy,
    #         max_features=1)
    
    # embs_manager = llm_tree.embed.EmbsManager()
    # test_tree_monotonic_in_depth(
    #     refinement_strategy='embs',
    #     max_features=1,
    #     embs_manager=embs_manager
    #     )
