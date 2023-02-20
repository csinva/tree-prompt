import tprompt.utils
import pickle as pkl
from os.path import join

def add_mean_llm_calls(r):
    mean_llm_calls = []
    for i, row in r.iterrows():
        if row.model_name == 'manual_tree':
            model = pkl.load(open(join(row.save_dir_unique, 'model.pkl'), 'rb'))
            mean_llm_calls.append(tprompt.utils.calculate_mean_depth_of_points_in_tree(model))
        elif 'num_prompts' in row:
            mean_llm_calls.append(row.num_prompts)
        else:
            mean_llm_calls.append(1)
    return mean_llm_calls