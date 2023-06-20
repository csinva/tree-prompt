from typing import List

from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import torch.cuda
repo_dir = dirname(dirname(os.path.abspath(__file__)))


# save_dir = '/home/chansingh/mntv1'
# save_dir = '/home/jxm3/research/prompting/tree-prompt/results'
save_dir = '/n/holyscratch01/rush_lab/Users/yuntian/tree/tree-prompt'

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    'save_dir': [join(save_dir, 'tree-prompt', 'jun20')],
    'cache_prompt_features_dir': ['/n/holyscratch01/rush_lab/Users/yuntian/tree/tree-prompt/cache_prompt_features'],
}

# List of tuples to sweep over (these values are coupled, and swept over together)
params_coupled_dict = {
    ('dataset_name', 'binary_classification', 
     'model_name', 'checkpoint', 'batch_size', 'num_prompts',
     'prompt_source', 'verbalizer_num', 'num_data_demonstrations_per_class'): [
        (dataset_name, binary_classification, 
         model_name, checkpoint, batch_size, num_prompts,
         prompt_source, verbalizer_num, num_data_demonstrations_per_class)

        for (checkpoint, batch_size) in [
            ('gpt2', 64),
            ('gpt2-medium', 32),
            ('gpt2-large', 32),
            ('gpt2-xl', 16),
            ('EleutherAI/gpt-j-6B', 8),  
        ]

        for (dataset_name, binary_classification) in [
            ('knnp__agnews', 0),
            ('knnp__cb', 0), 
            ('knnp__cr', 1), 
            ('knnp__dbpedia', 0), 
            ('knnp__mpqa', 1), 
            ('knnp__mr', 1), 
            ('knnp__rte', 1), 
            ('knnp__sst2', 1), 
            ('knnp__subj', 1), 
            ('knnp__trec', 0),
            ('rotten_tomatoes', 1),
            ('sst2', 1),
            ('imdb', 1),
            ('financial_phrasebank', 0),
            ('emotion', 0),
        ]

        for (model_name, num_prompts) in [
            for mod_name in ['manual_tree', 'manual_ensemble']
            for num_prompt in [40]
        ] # + [('manual_gbdt', 40), ('manual_rf', 40)]
        
    
        for (prompt_source, verbalizer_num, num_data_demonstrations_per_class) in [
            ('data_demonstrations', 0, 128),
        ]

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
    shuffle=False,
)
