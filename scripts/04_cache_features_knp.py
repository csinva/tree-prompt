from typing import List

from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import torch.cuda
import sys
repo_dir = dirname(dirname(os.path.abspath(__file__)))


datasets_with_binary = {
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
        # ('emotion', 0),
    ],
    4: [
        ('knnp__trec', 0),
        # ('sst2', 1),
    ],
    5: [
        ('knnp__cb', 0),
        # ('rotten_tomatoes', 1),
    ],
    6: [
        ('knnp__subj', 1),
        ('knnp__sst2', 1),
    ],
    # 7: [
    #     ('imdb', 1),
    #     ('financial_phrasebank', 0),
    # ]
}
datasets_with_binary_list = sum(list(datasets_with_binary.values()), [])

# save_dir = '/home/chansingh/mntv1'
# save_dir = '/home/jxm3/research/prompting/tree-prompt/results'
# save_dir = '/n/holyscratch01/rush_lab/Users/yuntian/tree/tree-prompt/results'
save_dir = "/home/chansingh/mntv1"
# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    "save_dir": [join(save_dir, "tree-prompt", "dummy_runs")],
    "cache_prompt_features_dir": [
        "/home/chansingh/mntv1/tree-prompt/cache_prompt_features"
    ],
    # 'save_dir': [join(save_dir, 'tree-prompt', 'jun22-full')],
    # 'cache_prompt_features_dir': ['/n/holyscratch01/rush_lab/Users/yuntian/tree/tree-prompt/cache_prompt_features'],
    'subsample_train': [10_000],
    "save_results": [0],
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
            for c in ["meta-llama/Llama-2-13b-hf"]
            # for c in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'EleutherAI/gpt-j-6B', "llama_7b", 'meta-llama/Llama-2-7b-hf']
            for b in [64, 16, 4, 1]
        ]

        for (dataset_name, binary_classification) in datasets_with_binary_list

        for (model_name, num_prompts) in [
            (mod_name, num_prompt)
            for mod_name in ['manual_tree']
            for num_prompt in [40]
        ]


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
    # gpu_ids=[0, 1, 2, 3],
    gpu_ids=[[0, 1], [2, 3]],
    shuffle=False,
)
