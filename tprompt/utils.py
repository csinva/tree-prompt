from sklearn.tree import DecisionTreeClassifier
import torch
import transformers
import numpy as np

def load_lm(
    checkpoint: str, 
    tokenizer: transformers.PreTrainedTokenizer, 
) -> transformers.AutoModelForCausalLM:
    kwargs = {
        'pretrained_model_name_or_path': checkpoint,
        'output_hidden_states': False,
        'pad_token_id': tokenizer.eos_token_id,
        'low_cpu_mem_usage': True,
    }
    if checkpoint == "EleutherAI/gpt-j-6B":
        print(f"loading model in fp16 from checkpoint {checkpoint}")
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            **kwargs,
            revision="float16", 
            torch_dtype=torch.float16, 
        )
    elif checkpoint.startswith('gpt2'):
        print(f"loading gpt model in fp16 from checkpoint {checkpoint}")
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            **kwargs,
            torch_dtype=torch.float16, 
        )
    else:
        print(f"loading model in fp32 from checkpoint {checkpoint}")
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            **kwargs,
        )
    return lm


def calculate_mean_depth_of_points_in_tree(clf: DecisionTreeClassifier):
    """Calculate the mean depth of each point in the tree.
    This is the average depth of the path from the root to the point.
    """
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right

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
    n_samples = clf.tree_.n_node_samples
    leaf_samples = n_samples[is_leaves]
    leaf_depths = node_depth[is_leaves]
    leaf_depths = leaf_depths.astype(np.float64)
    leaf_samples = leaf_samples.astype(np.float64)
    depths = leaf_depths * leaf_samples / np.sum(leaf_samples)
    return np.sum(depths)
