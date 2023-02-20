from typing import List

from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))


# save_dir = '/home/chansingh/mntv1'
save_dir = '/home/jxm3/research/prompting/tree-prompt/results'

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    'save_dir': [join(save_dir, 'tree-prompt', 'feb21')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
    # 'dataset_name': ['rotten_tomatoes', 'financial_phrasebank', 'emotion', 'sst2'],
    # 'dataset_name': ['rotten_tomatoes'], #, 'financial_phrasebank', 'emotion', 'sst2'],
    # 'dataset_name': ['imdb'],
    'dataset_name': ['rotten_tomatoes', 'sst2', 'imdb'],
    'verbalizer_num': [0], # [0, 1],
    'checkpoint': ['gpt2'], # gpt2? gpt2-xl?
}

# List of tuples to sweep over (these values are coupled, and swept over together)
params_coupled_dict = {
    # ('model_name', 'batch_size', 'num_prompts', 'prompt_source'): [
    #     (model_name, 4, num_prompts, prompt_source)
    #     for num_prompts in [1, 3, 5, 7, 10]
    #     for model_name in ['manual_ensemble', 'manual_tree', 'manual_boosting']
    #     for prompt_source in ['manual', 'data_demonstrations']
    # ],
    ('model_name', 'split_strategy', 'batch_size', 'max_depth',): [
        ('tprompt', 'iprompt', 32, max_depth) for max_depth in [1, 3, 5]
    ],
}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)

def get_gpu_ids() -> List[str]:
    """Get available GPUs on machine, but respect `CUDA_VISIBLE_DEVICES` env var."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    else:
        return list(range(torch.cuda.device_count()))

submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '01_fit.py'),
    actually_run=True,
    gpu_ids = get_gpu_ids(),
    shuffle=True,
    reverse=True,
)
