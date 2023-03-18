import tprompt.utils
import pickle as pkl
from os.path import join
import numpy as np
import joblib

def add_mean_llm_calls(r):
    mean_llm_calls = []
    for i, row in r.iterrows():
        if row.model_name == 'manual_tree':
            model = pkl.load(open(join(row.save_dir_unique, 'model.pkl'), 'rb'))
            mean_llm_calls.append(tprompt.utils.calculate_mean_depth_of_points_in_tree(model))
        elif row.model_name == 'manual_gbdt':
            # model = pkl.load(open(join(row.save_dir_unique, 'model.pkl'), 'rb'))
            # model = joblib.load(join(row.save_dir_unique, 'model.pkl'))
            # print(model.estimators_[0])
            # mean_calls_per_tree = [tprompt.utils.calculate_mean_depth_of_points_in_tree(tree[0]) for tree in model.estimators_]
            # mean_llm_calls.append(np.mean(mean_calls_per_tree))
            mean_llm_calls.append(25)
        elif row.model_name in ['manual_ensemble', 'manual_boosting']:
            mean_llm_calls.append(row.num_prompts)
        elif 'num_prompts' in row:
            mean_llm_calls.append(row.num_prompts)
        else:
            mean_llm_calls.append(1)
    return mean_llm_calls