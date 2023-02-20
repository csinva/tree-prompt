import os
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import sklearn
from tqdm import tqdm
import pandas as pd
import pickle as pkl
# import notebook_helper
import dtreeviz
import dtreeviz.trees
from sklearn.tree import plot_tree
import sys
import numpy as np
import llm_tree.data
import dvu
import dtreeviz
import numpy as np
import imodelsx.viz
import pandas as pd
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree

from os.path import join, dirname
path_to_repo = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(join(path_to_repo, 'experiments/'))
from collections import namedtuple

from sklearn import __version__
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree

DSETS_RENAME_DICT = {
    'emotion': 'Emotion',
    'sst2': 'SST2',
    'tweet_eval': 'Tweet (Hate)',
    'rotten_tomatoes': 'Rotten tomatoes',
    'financial_phrasebank': 'Financial phrasebank',
}

MODELS_RENAME_DICT = {
    'llm_tree': 'LLM-Tree',
    'id3': 'ID3',
    'decision_tree': 'CART',
}

XLAB = {
    'max_depth': 'Tree depth',
    'n_estimators': '# estimators',
    'mean_llm_calls': '# LLM calls',
    'num_prompts': '# prompts',
}

def plot_perf_curves_individual(rp, x='max_depth', fname_save='../results/figs/perf_curves_individual.pdf'):
    dset_names = rp['dataset_name'].unique()
    R, C = 1, min(3, len(dset_names))
    plt.figure(figsize=(C * 2.5, R * 2.5))
    for i in range(R * C):
        plt.subplot(R, C, i + 1)
        dset_name = dset_names[i]
        rd = rp[rp.dataset_name == dset_name]
        groupings = 'model_name'
        for (k, g) in rd.groupby(by=groupings):
            # print(g.columns)

            if 'llm_tree' in k:
                kwargs = {'lw': 1.5, 'alpha': 0.9, 'ls': '-', 'marker': '.', 'color': 'black'}
            else:
                kwargs = {'alpha': 0.5, 'lw': 1, 'ls': '--', 'marker': '.'}

            plt.plot(g[x], g['roc_auc_test'], label=MODELS_RENAME_DICT.get(k, k), **kwargs)
        plt.title(DSETS_RENAME_DICT.get(dset_name, dset_name), fontsize='medium')
        plt.xlabel(XLAB.get(x, x))
        plt.ylabel('ROC AUC')
    plt.legend(
        title_fontsize='xx-small',
        labelcolor='linecolor',
        # bbox_to_anchor=(1.5, 1.1),
        fontsize='x-small'
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(fname_save), exist_ok=True)
    plt.savefig(fname_save, bbox_inches='tight')
    plt.show()

def plot_train_and_test(ravg, groupings, metric, x='max_depth'):
    for dataset_name in ravg['dataset_name'].unique():
        ravg_dset = ravg[ravg['dataset_name'] == dataset_name]
    
        plt.figure(figsize=(8, 3))
        for (k, g) in ravg_dset.groupby(by=groupings):
            if 'llm_tree' in k:
                kwargs = {'lw': 1.5, 'alpha': 0.5, 'ls': '-', 'marker': '.'}
            else:
                kwargs = {'alpha': 0.5, 'lw': 1, 'ls': '--', 'marker': '.'}
            lab = k
            
            R, C = 1, 2

            plt.subplot(R, C, 1)
            plt.plot(g[x], g[metric + '_train'], label=lab, **kwargs)
            plt.ylabel(f'{metric} (train)')
            plt.xlabel(XLAB.get(x, x))
            plt.xscale('log')

            plt.subplot(R, C, 2)
            plt.plot(g[x], g[metric + '_test'], label=lab, **kwargs)
            plt.ylabel(f'{metric} (test)')
            plt.xlabel(XLAB.get(x, x))
            plt.xscale('log')
            plt.title(dataset_name, fontsize='small')

        for i in range(1, C+1):
            plt.subplot(R, C, i)
            plt.grid()

        plt.legend(
            title=',\n'.join(groupings),
            title_fontsize='xx-small',
            labelcolor='linecolor',
            bbox_to_anchor=(1.5, 1.1), fontsize='x-small'
        )
        # plt.tight_layout()
        # dvu.line_legend()

        # plt.show()

def save_tree(model, X_train, y_train, feature_names,
            target_name='y',
            class_names=["neg", "pos"], fname='tree.svg'):
    viz_model = dtreeviz.model(
        model,
        X_train=X_train,
        y_train=np.array(y_train),
        feature_names=np.array(feature_names),
        target_name=target_name,
        class_names=np.array(class_names)
    )
    v = viz_model.view()
    v.show() # don't call this, opens a pop-up
    v.save(fname)
    # plt.close()



if __name__ == '__main__':
    results_dir = join(path_to_repo, 'results/feb11/')
    r = notebook_helper.get_results_df(results_dir, use_cached=True)
    rd = r[r.dataset_name == 'sst2']
    rd = rd[rd.max_depth == 3]
    
    
    run_args = rd[(rd.model_name == 'llm_tree')].sort_values(by='accuracy_cv').iloc[-1]
    # run_args = rd[(rd.model_name == 'decision_tree')].sort_values(by='accuracy_cv').iloc[-1]
    model = pkl.load(open(join(run_args.save_dir_unique, 'model.pkl'), 'rb'))
    print('acc', run_args.accuracy_test, 'depth', run_args.max_depth)

    X_train, X_cv, X_test, y_train, y_cv, y_test, feature_names = \
    llm_tree.data.get_all_data(run_args)

    dt, strs_array = imodelsx.viz.extract_sklearn_tree_from_llm_tree(model, n_classes=2, dtreeviz_dummies=False)
    sklearn.tree.plot_tree(dt, feature_names=strs_array, class_names=['neg', 'pos'],
                          precision=2, rounded=True)
    plt.savefig('tree.pdf')
    plt.close()

    dt, strs_array = imodelsx.viz.extract_sklearn_tree_from_llm_tree(model, n_classes=2, dtreeviz_dummies=True)
    shadow_dtree = ShadowSKDTree(
        dt, X_train.toarray(), y_train, strs_array, 'y', [0, 1])
    
    viz_model = dtreeviz.trees.dtreeviz(shadow_dtree)
    viz_model.save('tree.svg')
    plt.close()
    

    # for DT
    # viz.save_tree(model, X_train.toarray(), y_train, feature_names, fname='tree.svg')
    # viz_model = dtreeviz.model(
    #     model,
    #     X_train=X_train.toarray(),
    #     y_train=np.array(y_train),
    #     feature_names=np.array(feature_names),
    #     target_name='y',
    #     class_names=np.array(['neg', 'pos'])
    # )
    # v = viz_model.view()
    # v.show() # don't call this, opens a pop-up
    # v.save('tree.svg')
    # )

    print('succesfully saved!')