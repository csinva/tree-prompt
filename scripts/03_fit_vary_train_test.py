from typing import List

from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import torch.cuda

repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/tree-prompt/experiments/01_fit.py --dataset_name rotten_tomatoes --binary_classification 1 --model_name manual_gbdt --checkpoint EleutherAI/gpt-j-6B --checkpoint_evaluation EleutherAI/gpt-j-6B --prompt_source manual --verbalizer_num 0 --seed 1 --save_dir /home/chansingh/mntv1/tree-prompt/aug22_vary_train_test --cache_prompt_features_dir /home/chansingh/mntv1/tree-prompt/cache_prompt_features --num_data_demonstrations_per_class 1 --num_prompts 40 
# python /home/chansingh/tree-prompt/experiments/01_fit.py --dataset_name rotten_tomatoes --binary_classification 1 --model_name manual_tree --checkpoint gpt2 --checkpoint_evaluation gpt2 --prompt_source manual --verbalizer_num 0 --seed 1 --save_dir /home/chansingh/mntv1/tree-prompt/aug22_vary_train_test --cache_prompt_features_dir /home/chansingh/mntv1/tree-prompt/cache_prompt_features --num_data_demonstrations_per_class 1 --num_prompts 40 
save_dir = "/home/chansingh/mntv1"

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    "seed": [1],
    "save_dir": [join(save_dir, "tree-prompt", "aug22_vary_train_test")],
    "cache_prompt_features_dir": [
        "/home/chansingh/mntv1/tree-prompt/cache_prompt_features"
    ],
    "num_data_demonstrations_per_class": [1],
    # "subsample_frac": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    "num_prompts": [40],
}

# List of tuples to sweep over (these values are coupled, and swept over together)
params_coupled_dict = {
    (
        "dataset_name",
        "binary_classification",
        "model_name",
        "checkpoint",
        "checkpoint_evaluation",
        "prompt_source",
        "verbalizer_num",
    ): [
        (
            dataset_name,
            binary_classification,
            model_name,
            checkpoint,
            checkpoint_evaluation,
            prompt_source,
            verbalizer_num,
        )
        for (checkpoint, checkpoint_evaluation) in [
            ("gpt2", "gpt2"),
            ("gpt2", "EleutherAI/gpt-j-6B"),
            ("EleutherAI/gpt-j-6B", "EleutherAI/gpt-j-6B"),
            ("EleutherAI/gpt-j-6B", "gpt2"),
        ]
        for (dataset_name, binary_classification) in [
            ("rotten_tomatoes", 1),
            ("sst2", 1),
            ("imdb", 1),
            ("financial_phrasebank", 0),
            ("emotion", 0),
        ]
        for model_name in [
            "manual_tree",
            "manual_gbdt",
        ]
        for (prompt_source, verbalizer_num) in [
            ("manual", 0),
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
    script_name=join(repo_dir, "experiments", "01_fit.py"),
    actually_run=True,
    # n_cpus=16,
    # gpu_ids = get_gpu_ids(),
    reverse=True,
    n_cpus=8,
    shuffle=False,
)
