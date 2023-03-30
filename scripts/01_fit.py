from typing import List

from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import torch.cuda
repo_dir = dirname(dirname(os.path.abspath(__file__)))


save_dir = '/home/chansingh/mntv1'
# save_dir = '/home/jxm3/research/prompting/tree-prompt/results'

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    # 'save_dir': [join(save_dir, 'tree-prompt', 'feb25')],
    'save_dir': [join(save_dir, 'tree-prompt', 'mar18')],
    # 'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
    # 'dataset_name': ['rotten_tomatoes', 'sst2'], #, 'imdb'],
    # 'verbalizer_num': [0], # [0, 1],
    'cache_prompt_features_dir': ['/home/chansingh/mntv1/tree-prompt/cache_prompt_features'],
}

# List of tuples to sweep over (these values are coupled, and swept over together)
params_coupled_dict = {
    ('dataset_name', 'binary_classification', 
     'model_name', 'checkpoint', 'batch_size', 'num_prompts',
     'prompt_source', 'verbalizer_num'): [
        (dataset_name, binary_classification, 
         model_name, checkpoint, batch_size, num_prompts,
         prompt_source, verbalizer_num)

        for (checkpoint, batch_size) in [
            ('gpt2', 4),
            # ('EleutherAI/gpt-j-6B', 2),
            ('EleutherAI/gpt-j-6B', 1),  
        ]

        for (dataset_name, binary_classification) in [
            ('rotten_tomatoes', 1),
            ('sst2', 1),
            # ('imdb', 1),
            ('financial_phrasebank', 0),
            ('emotion', 0),
        ]

        for (model_name, num_prompts) in [
            (mod_name, num_prompt)
            for mod_name in ['manual_ensemble', 'manual_tree', 'manual_boosting']
            for num_prompt in [1, 3, 5, 7, 10, 15, 25, 40]
        ] + [('manual_gbdt', 40), ('manual_rf', 40)]
        
    
        for (prompt_source, verbalizer_num) in [
            ('manual', 0),
            ('data_demonstrations', 0),
            ('data_demonstrations', 1)
        ]

    ],
    
    # ('model_name', 'split_strategy', 'max_depth',): [
    #     ('tprompt', 'iprompt', max_depth)
    #    for max_depth in [1, 3, 5]),
    # ('model_name', 'batch_size', 'num_prompts', 'prompt_source'): [
    #     (model_name, 4, num_prompts, prompt_source)
    #     for num_prompts in [1, 3, 5, 7, 10]
    #     for model_name in ['manual_ensemble', 'manual_tree', 'manual_boosting']
    #     for prompt_source in ['manual', 'data_demonstrations']
    # ],
    # ('model_name', 'split_strategy', 'batch_size', 'max_depth',): [
        # ('tprompt', 'iprompt', 32, max_depth) for max_depth in [1, 3, 5]
    # ],
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
    # n_cpus=16,
    # gpu_ids = get_gpu_ids(),
    # gpu_ids=[0, 1],
    # n_cpus=4,
    # gpu_ids = [0, 1, 2, 3],
    # gpu_ids = [0],
    # reverse=True,
    n_cpus=4,
    shuffle=False,
)
