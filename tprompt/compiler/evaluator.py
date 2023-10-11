from functools import partial
import imodelsx.treeprompt.stump
from sklearn.preprocessing import OneHotEncoder
import sklearn.tree
import joblib
from dict_hash import sha256
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import imodelsx.process_results
from collections import defaultdict
import numpy as np
import transformers
import sys
import tprompt.utils
from os.path import join
import datasets
from typing import Dict, List
from sklearn.tree import plot_tree

from abc import ABC, abstractmethod
import logging
import math
import random
import imodels
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


def modify_activations(module, inputs, outputs, hook_weights=None, prompt_at_start_or_end="end"):
    if hook_weights is not None:
        hook_weights = hook_weights.repeat((outputs.shape[0], 1, 1))

        # hacky fix for rare err -- sometimes prompt is longer than sequence
        # weird because sequence should include the prompt
        # Cause: tokenization issue merges a couple tokens
        # keep only the end of the prompt
        seq_len = min(hook_weights.shape[1], outputs.shape[1])
        hook_weights = hook_weights[:, -seq_len:, :]

        if prompt_at_start_or_end == "end":
            outputs[:, -seq_len:, :] = hook_weights
        elif prompt_at_start_or_end == "start":
            outputs[:, :seq_len, :] = hook_weights

    return outputs


class PromptHooker(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        checkpoint: str,
        prompts: List[str],
        verbalizer: Dict[int, str] = {0: " Negative.", 1: " Positive."},
        batch_size: int = 1,
        prompt_template: str = "{example}{prompt}",
        cache_prompt_features_dir=join("cache_prompt_features"),
        cache_key_values: bool = False,
        device=None,
        verbose: bool = True,
        random_state: int = 42,
        hook_weights=None,  # torch.Tensor
        prompt_at_start_or_end: str = "end",
    ):
        '''
        Params
        ------
        prompt_template: str
            template for the prompt, for different prompt styles (e.g. few-shot), may want to place {prompt} before {example}
             or you may want to add some text before the verbalizer, e.g. {example}{prompt} Output:
        cache_key_values: bool
            Whether to cache key values (only possible when prompt does not start with {example})
        '''
        self.checkpoint = checkpoint
        self.prompts = prompts
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.prompt_template = prompt_template
        self.cache_prompt_features_dir = cache_prompt_features_dir
        self.cache_key_values = cache_key_values
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        self.hook_weights = hook_weights
        global HOOK_WEIGHTS
        HOOK_WEIGHTS = hook_weights
        self.prompt_at_start_or_end = prompt_at_start_or_end
        assert prompt_at_start_or_end in ["start", "end"]

    def fit(self, X, y):
        transformers.set_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        assert len(self.classes_) == len(self.verbalizer)

        # calculate prompt features
        prompt_features = self._calc_prompt_features(X, self.prompts)
        self.prompt_accs_ = [
            accuracy_score(y, prompt_features[:, i]) for i in range(len(self.prompts))
        ]

        return self

    def _calc_prompt_features(self, X, prompts):
        prompt_features = np.zeros((len(X), len(prompts)))
        llm = imodelsx.llm.get_llm(self.checkpoint)._model
        if self.device is not None:
            llm = llm.to(self.device)

        if self.hook_weights is not None:
            # register a forward hook on the "drop" layer
            hook = llm.transformer.drop.register_forward_hook(
                partial(modify_activations, hook_weights=self.hook_weights,
                        prompt_at_start_or_end=self.prompt_at_start_or_end))
            self.hook_weights = self.hook_weights.to(llm.device)

        stump = None

        for i, prompt in enumerate(prompts):
            if self.verbose:
                print(f"Prompt {i}: {prompt}")

            loaded_from_cache = False
            if self.cache_prompt_features_dir is not None:
                os.makedirs(self.cache_prompt_features_dir, exist_ok=True)
                args_dict_cache = {"prompt": prompt,
                                   "X_len": len(X), "ex0": X[0]}
                save_dir_unique_hash = sha256(args_dict_cache)
                cache_file = join(
                    self.cache_prompt_features_dir, f"{save_dir_unique_hash}.pkl"
                )

                # load from cache if possible
                if os.path.exists(cache_file):
                    if self.verbose:
                        print("loading from cache!")
                    try:
                        prompt_features_i = joblib.load(cache_file)
                        loaded_from_cache = True
                    except:
                        pass

            if not loaded_from_cache:
                if stump is None:
                    stump = imodelsx.treeprompt.stump.PromptStump(
                        model=llm,
                        checkpoint=self.checkpoint,
                        verbalizer=self.verbalizer,
                        batch_size=self.batch_size,
                        prompt_template=self.prompt_template,
                        cache_key_values=self.cache_key_values,
                    )

                # calculate prompt_features
                def _calc_features_single_prompt(
                    X, stump, prompt, past_key_values=None
                ):
                    """Calculate features with a single prompt (results get cached)
                    preds: np.ndarray[int] of shape (X.shape[0],)
                        If multiclass, each int takes value 0, 1, ..., n_classes - 1 based on the verbalizer
                    """
                    stump.prompt = prompt
                    if past_key_values is not None:
                        preds = stump.predict_with_cache(X, past_key_values)
                    else:
                        preds = stump.predict(X)
                    return preds

                past_key_values = None
                if self.cache_key_values:
                    stump.prompt = prompt
                    past_key_values = stump.calc_key_values(X)
                prompt_features_i = _calc_features_single_prompt(
                    X, stump, prompt, past_key_values=past_key_values
                )
                if self.cache_prompt_features_dir is not None:
                    joblib.dump(prompt_features_i, cache_file)

            # save prompt features
            prompt_features[:, i] = prompt_features_i

        # remove hook
        if self.hook_weights is not None:
            hook.remove()

        return prompt_features
