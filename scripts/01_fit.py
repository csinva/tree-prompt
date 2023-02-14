from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    'save_dir': [join('/home/chansingh/mntv1', 'tree-prompt', 'feb14')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
    'dataset_name': ['rotten_tomatoes', 'financial_phrasebank', 'emotion', 'sst2'],
    'max_depth': [1, 3, 5],
    'split_strategy': ['iprompt'],
    'verbalizer_num': [0, 1],
}

# List of tuples to sweep over (these values are coupled, and swept over together)
params_coupled_dict = {}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '01_fit.py'),
    actually_run=True,
    gpu_ids = [0],
    shuffle=False,
)
