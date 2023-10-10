from evaluator import PromptHooker, modify_activations
import imodelsx.treeprompt.stump
from sklearn.preprocessing import OneHotEncoder
import sklearn.tree
import random
import joblib
from dict_hash import sha256
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import imodelsx.process_results
from collections import defaultdict
import numpy as np
from copy import deepcopy
import viz
import transformers
import sys
import tprompt.utils
from os.path import join
import datasets
from typing import Dict, List
from sklearn.tree import plot_tree
import imodelsx.util
import imodelsx.metrics
import numpy as np
import tprompt.utils
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import torch.cuda
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
sys.path.append('../experiments/')

OUTPUTS_ALL = {}
PROMPT_NUM_GLOBAL = 0


def get_avg_soft_prompt(checkpoint, prompts):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).eval()

    def store_activations(module, inputs, outputs):
        global OUTPUTS_ALL
        global PROMPT_NUM_GLOBAL
        OUTPUTS_ALL[PROMPT_NUM_GLOBAL] = outputs.detach().cpu()
        PROMPT_NUM_GLOBAL += 1
        return outputs

    hook = model.transformer.drop.register_forward_hook(store_activations)
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        # hook = model.transformer.h[3].register_forward_hook(change_activations)
        _ = model(**inputs)

    hook.remove()
    assert len(OUTPUTS_ALL) == len(prompts)

    # most_probable_tokens = torch.topk(logits_modified, k=10, dim=-1)
    # print('\n'.join([tokenizer.decode(x)
    #   for x in most_probable_tokens.indices[0, -1]]))
    # logits_orig = model(**inputs).logits

    vals = list(OUTPUTS_ALL.values())
    emb_size = vals[0].shape[-1]

    max_len = max([x.shape[1] for x in vals])
    # add left padding
    padded = [torch.cat([torch.zeros((1, max_len - x.shape[1], emb_size)), x], dim=1)
              for x in vals]

    # average
    avg = torch.concat(tuple(padded)).mean(axis=0).unsqueeze(0)
    return avg
