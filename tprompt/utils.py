from sklearn.tree import DecisionTreeClassifier
import torch
import transformers
import numpy as np
import pickle as pkl
from os.path import join
import numpy as np
import joblib
import os.path
from transformers import AutoTokenizer
import os


LLAMA_DIR = os.path.expanduser("~/llama")  # expects a folder in here named 'llama_7b'

os.environ["LLAMA_TOKEN"] = "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"


def load_tokenizer(checkpoint: str) -> transformers.PreTrainedTokenizer:
    if ('llama' in checkpoint.lower()) and ('llama-2' not in checkpoint.lower()):
        return transformers.LlamaTokenizer.from_pretrained(join(LLAMA_DIR, checkpoint))
    else:
        return AutoTokenizer.from_pretrained(checkpoint, 
            token=os.environ.get("LLAMA_TOKEN"),)


def load_lm(
    checkpoint: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> transformers.AutoModelForCausalLM:
    kwargs = {
        "pretrained_model_name_or_path": checkpoint,
        "output_hidden_states": False,
        "pad_token_id": tokenizer.eos_token_id,
        "low_cpu_mem_usage": True,
    }
    if checkpoint == "EleutherAI/gpt-j-6B":
        print(f"loading model in fp16 from checkpoint {checkpoint}")
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            **kwargs,
            revision="float16",
            torch_dtype=torch.float16,
        )
    elif "llama-2" in checkpoint.lower():
        print("loading llama model:", checkpoint)
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.environ.get("LLAMA_TOKEN"),
            offload_folder="offload",
        )   
    elif checkpoint.startswith("gpt2") and not checkpoint == "gpt2":
        print(f"loading gpt model in fp16 from checkpoint {checkpoint}")
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            **kwargs,
            torch_dtype=torch.float16,
        )
    elif checkpoint.startswith("llama_"):
        lm = transformers.LlamaForCausalLM.from_pretrained(
            join(LLAMA_DIR, checkpoint),
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        print(f"loading model in fp32 from checkpoint {checkpoint}")
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            **kwargs,
        )
    return lm


def calculate_mean_depth_of_points_in_tree(tree_):
    """Calculate the mean depth of each point in the tree.
    This is the average depth of the path from the root to the point.
    """
    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    # iterate over leaves and calculate the number of samples in each of them
    n_samples = tree_.n_node_samples
    leaf_samples = n_samples[is_leaves]
    leaf_depths = node_depth[is_leaves]
    leaf_depths = leaf_depths.astype(np.float64)
    leaf_samples = leaf_samples.astype(np.float64)
    depths = leaf_depths * leaf_samples / np.sum(leaf_samples)
    return np.sum(depths)


def calculate_mean_unique_calls_in_ensemble(ensemble, X):
    if X is None:
        # Should pass X, this is just for testing
        n_features_in = ensemble.n_features_in_
        X = np.random.randint(2, size=(100, n_features_in))

    # extract the decision path for each sample
    ests = ensemble.estimators_.flatten()
    feats = [set() for _ in range(len(X))]
    for i in range(len(ests)):
        est = ests[i]
        node_index = est.decision_path(X).toarray()
        feats_est = [
            set([est.tree_.feature[x] for x in np.nonzero(row)[0]])
            for row in node_index
        ]
        for j in range(len(feats)):
            feats[j] = feats[j].union(feats_est[j])
    # -1 for the -2 feature that is always present
    return np.mean([len(f) - 1 for f in feats])


def add_mean_llm_calls(r):
    mean_llm_calls = []
    for i, row in r.iterrows():
        if row.model_name in ["manual_tree", "manual_gbdt", "manual_hstree"]:
            model = pkl.load(open(join(row.save_dir_unique, "model.pkl"), "rb"))
        else:
            model = None
        mean_llm_calls.append(
            compute_mean_llm_calls(row.model_name, row.num_prompts, model=model)
        )
    return mean_llm_calls


def compute_mean_llm_calls(model_name, num_prompts, model=None, X=None):
    if model_name == "manual_tree":
        return calculate_mean_depth_of_points_in_tree(model.tree_)
    elif model_name == "manual_hstree":
        return calculate_mean_depth_of_points_in_tree(model.estimator_.tree_)
    elif model_name == "manual_gbdt":
        return calculate_mean_unique_calls_in_ensemble(model, X)
    elif model_name == "manual_tree_cv":
        return calculate_mean_depth_of_points_in_tree(model.best_estimator_.tree_)
    elif model_name in ["manual_single_prompt"]:
        return 1
    elif model_name in ["manual_ensemble", "manual_boosting"]:
        return num_prompts
    else:
        return num_prompts
