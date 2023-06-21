from typing import List

from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import torch.cuda
import sys
repo_dir = dirname(dirname(os.path.abspath(__file__)))


assert len(sys.argv) == 2, "need to run like `python scripts/01_fit_knp.py <run_id>` where run_id in [0, 7]"
run_id = int(sys.argv[1])
assert run_id in range(0, 8), f"invalid run_id {run_id}, must be in [0, 1, .., 7]"


datasets_for_run_id = {
        0: [
            ('knnp__agnews', 0),
            ('knnp__cr', 1), 
        ],
        1: [
            ('knnp__dbpedia', 0), 
        ],
        2: [
            ('knnp__mpqa', 1), 
            ('knnp__mr', 1), 
        ],
        3: [
            ('knnp__rte', 1), 
            ('emotion', 0),
        ],
        4: [
            ('knnp__trec', 0),
        ],
        5: [
            ('knnp__cb', 0), 
            ('rotten_tomatoes', 1),
        ],
        6: [
            ('knnp__subj', 1), 
            ('knnp__sst2', 1), 
            ('sst2', 1),
        ],
        7: [
            ('imdb', 1),
            ('financial_phrasebank', 0),
        ]
    }


# save_dir = '/home/chansingh/mntv1'
# save_dir = '/home/jxm3/research/prompting/tree-prompt/results'
save_dir = '/n/holyscratch01/rush_lab/Users/yuntian/tree/tree-prompt/results'

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1, 2, 3],
    'save_dir': [join(save_dir, 'tree-prompt', 'jun21')],
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
            (c, b) 
            for c in [('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'EleutherAI/gpt-j-6B')]
            for b in [128, 64, 32, 16, 4, 1]
        ]

        for (dataset_name, binary_classification) in datasets_for_run_id[run_id]

        for (model_name, num_prompts) in [
            (mod_name, num_prompt)
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
