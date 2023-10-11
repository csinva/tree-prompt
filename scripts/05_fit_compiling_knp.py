from typing import List

from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import torch.cuda
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# python /home/chansingh/tree-prompt/experiments/01_fit.py --dataset_name emotion --binary_classification 0 --model_name manual_ensemble --checkpoint gpt2 --batch_size 2 --num_prompts 15 --prompt_source data_demonstrations --verbalizer_num 0 --num_data_demonstrations_per_class 1 --seed 1 --save_dir /home/chansingh/test --cache_prompt_features_dir /home/chansingh/mntv1/tree-prompt/cache_prompt_features
save_dir = '/home/chansingh/mntv1'

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    'save_dir': [join(save_dir, 'compiling', 'oct10')],
    'prompt_at_start_or_end': ['start'],
    'cache_prompt_features_dir': ['/home/chansingh/features_compiling'],
    'subsample_train_size': [300],

    'num_prompts': [80],
    'filter_by_median': [10],
}

params_coupled_dict = {
    ('dataset_name', 'binary_classification',
     'model_name', 'checkpoint', 'batch_size',
     'prompt_source', 'verbalizer_num', 'num_data_demonstrations_per_class'): [
        (dataset_name, binary_classification,
         model_name, checkpoint, batch_size,
         prompt_source, verbalizer_num, num_data_demonstrations_per_class)

        for (checkpoint, batch_size) in [
            (c, b)
            for c in ['gpt2', 'gpt2-large']
            # for c in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'EleutherAI/gpt-j-6B']
            # for c in ['meta-llama/Llama-2-7b-hf']
            for b in [64, 16, 4, 1]
        ]

        for (dataset_name, binary_classification) in [
            # ('knnp__agnews', 0),
            ('knnp__cr', 1),
            # ('knnp__dbpedia', 0),
            ('knnp__mpqa', 1),
            # ('knnp__mr', 1),
            ('knnp__rte', 1),
            ('emotion', 0),
            ('knnp__trec', 0),
            # ('sst2', 1),
            ('knnp__cb', 0),
            ('rotten_tomatoes', 1),
            ('knnp__subj', 1),
            # ('knnp__sst2', 1),
            # ('imdb', 1),
            # ('financial_phrasebank', 0),
        ]

        for model_name in ['single_prompt']


        for (prompt_source, verbalizer_num, num_data_demonstrations_per_class) in [
            ('data_demonstrations', 0, 4),  # was 128
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


# args_list = args_list[:1]
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '02_fit_compiling.py'),
    actually_run=True,
    # n_cpus=16,
    # gpu_ids=get_gpu_ids(),
    # gpu_ids=[0, 1, 2, 3],
    # n_cpus=4,
    gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[0],
    # reverse=True,
    # n_cpus=4,
    shuffle=False,
)
