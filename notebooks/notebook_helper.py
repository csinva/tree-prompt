import tprompt.utils
import pickle as pkl
from os.path import join
import numpy as np
import joblib


def add_mean_llm_calls(r):
    mean_llm_calls = []
    for i, row in r.iterrows():
        model = pkl.load(open(join(row.save_dir_unique, "model.pkl"), "rb"))
        mean_llm_calls.append(
            compute_mean_llm_calls(row.model_name, row.num_prompts, model=model)
        )
    return mean_llm_calls


def compute_mean_llm_calls(model_name, num_prompts, model=None):
    if model_name == "manual_tree":
        return tprompt.utils.calculate_mean_depth_of_points_in_tree(model)
    elif model_name == "manual_gbdt":
        # model = pkl.load(open(join(row.save_dir_unique, 'model.pkl'), 'rb'))
        # model = joblib.load(join(row.save_dir_unique, 'model.pkl'))
        # print(model.estimators_[0])
        # mean_calls_per_tree = [tprompt.utils.calculate_mean_depth_of_points_in_tree(tree[0]) for tree in model.estimators_]
        # mean_llm_calls.append(np.mean(mean_calls_per_tree))
        return 25
    elif model_name in ["manual_single_prompt"]:
        return 1
    elif model_name in ["manual_ensemble", "manual_boosting"]:
        return num_prompts
    else:
        return num_prompts
