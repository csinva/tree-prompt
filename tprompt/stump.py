from typing import Dict, List

from abc import ABC, abstractmethod
import logging

import imodels
import imodelsx.util
import imodelsx.metrics
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import torch.cuda
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class Stump(ABC):
    def __init__(
        self,
        args,
        split_strategy: str = "iprompt",
        tokenizer=None,
        max_features=10,
        assert_checks: bool = False,
        verbose: bool = True,
        model: AutoModelForCausalLM = None,
        checkpoint: str = "EleutherAI/gpt-j-6B",
        checkpoint_prompting: str = "EleutherAI/gpt-j-6B",
        verbalizer: Dict[str] = {0: " Negative.", 1: " Positive."},
    ):
        """Fit a single stump.
        Can use tabular features...
            Currently only supports binary classification with binary features.
        Params
        ------
        split_strategy: str
            'iprompt' - use iprompt to split
            'manual' - use passed prompt in args.prompt
            'cart' - use cart to split
            'linear' - use linear to split
        max_features: int
            used by StumpTabular to decide how many features to save
        checkpoint: str
            the underlying model used for prediction
        model: AutoModelForCausalLM
            if this is passed, will override checkpoint
        checkpoint_prompting: str
            the model used for finding the prompt
        """
        self.args = args
        assert split_strategy in ["iprompt", "cart", "linear", "manual"]
        self.split_strategy = split_strategy
        self.assert_checks = assert_checks
        self.verbose = verbose
        self.max_features = max_features
        self.checkpoint = checkpoint
        self.checkpoint_prompting = checkpoint_prompting
        self.model = model
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        else:
            self.tokenizer = tokenizer

        # tree stuff
        self.child_left = None
        self.child_right = None

    def __getstate__(self):
        """Get the stump but prevent certain attributes from being pickled.

        See also https://stackoverflow.com/a/54139237/2287177
        """
        state = self.__dict__.copy()
        # Don't pickle big things
        if "model" in state:
            del state["model"]
        if "tokenizer" in state:
            del state["tokenizer"]
        if "feature_names" in state:
            del state["feature_names"]
        return state

    @abstractmethod
    def fit(self, X_text: List[str], y: List[int], feature_names=None, X=None):
        return self

    @abstractmethod
    def predict(self, X_text: List[str]) -> np.ndarray[int]:
        return

    def _set_value_acc_samples(self, X_text, y):
        """Set value and accuracy of stump."""
        idxs_right = self.predict(X_text).astype(bool)
        n_right = idxs_right.sum()
        if n_right == 0 or n_right == y.size:
            self.failed_to_split = True
            return
        else:
            self.failed_to_split = False
        self.value = [np.mean(y[~idxs_right]), np.mean(y[idxs_right])]
        self.value_mean = np.mean(y)
        self.n_samples = [y.size - idxs_right.sum(), idxs_right.sum()]
        self.acc = accuracy_score(y, 1 * idxs_right)


class PromptStump(Stump):
    def __init__(self, *args, **kwargs):
        super(PromptStump, self).__init__(*args, **kwargs)
        if self.verbose:
            logging.info(f"Loading model {self.checkpoint}")

    def fit(self, X_text: List[str], y, feature_names=None, X=None):
        # check input and set some attributes
        assert len(np.unique(y)) > 1, "y should have more than 1 unique value"
        X, y, _ = imodels.util.arguments.check_fit_arguments(self, X, y, feature_names)
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # actually run fitting
        input_strings = X_text
        verbalizer_dict = self._get_verbalizer()
        output_strings = [verbalizer_dict[int(yi)] for yi in y]

        # get prompt
        if self.split_strategy == "manual":
            self.prompt = self.args.prompt
        else:
            # self.model = self.model.to('cpu')
            print(
                f"calling explain_dataset_iprompt with batch size {self.args.batch_size}"
            )
            prompts, metadata = imodelsx.explain_dataset_iprompt(
                lm=self.model,
                input_strings=input_strings,
                output_strings=output_strings,
                checkpoint=self.checkpoint,  # which language model to use
                num_learned_tokens=12,  # how long of a prompt to learn
                n_shots=1,  # number of examples in context
                n_epochs=5,  # how many epochs to search
                batch_size=self.args.batch_size,  # batch size for iprompt
                llm_float16=False,  # whether to load the model in float_16
                verbose=1,  # how much to print
                # sets template like ${input}${prefix}
                prefix_before_input=False,
                mask_possible_answers=True,  # only compute loss over valid output tokens
                generation_repetition_penalty=1.0,
                pop_topk_strategy="different_start_token",
                pop_criterion="acc",
                max_n_datapoints=len(input_strings),
                # on an a6000 gpu with gpt2-xl in fp16 and batch size 32,
                # 100 steps takes around 30 minutes.
                max_n_steps=1000,  # limit search by a fixed number of steps
            )
            torch.cuda.empty_cache()
            # self.model = self.model.to(self.device)

            # save stuff
            self.prompt = prompts[0]
            print(f"Got {len(prompts)} prompts. Top prompt: `{prompts[0]}`")
            self.prompts = prompts
            self.meta = metadata

        # set value (calls self.predict)
        self._set_value_acc_samples(X_text, y)

        return self

    def predict(self, X_text: List[str]) -> np.ndarray[int]:
        preds_proba = self.predict_proba(X_text)
        return np.argmax(preds_proba, axis=1)

    def predict_proba(self, X_text: List[str]) -> np.ndarray[float]:
        target_strs = list(self._get_verbalizer().values())

        # only predict based on first token of output string
        target_token_ids = list(map(self._get_first_token_id, target_strs))
        if self.args.prompt_source == "data_demonstrations":
            template = self.args.template_data_demonstrations
            preds = self._get_logit_for_target_tokens_batched(
                [self.prompt + template % (x, "") for x in X_text],
                target_token_ids,
                batch_size=self.args.batch_size,
            )
        else:
            preds = self._get_logit_for_target_tokens_batched(
                [x + self.prompt for x in X_text],
                target_token_ids,
                batch_size=self.args.batch_size,
            )
        # preds = np.zeros((len(X_text), len(target_token_ids)))
        # for i, x in enumerate(X_text):
        #     preds[i] = self._get_logit_for_target_tokens(x, target_token_ids)
        #     preds[i] = self._get_logit_for_target_tokens(x + self.prompt, target_token_ids)
        assert preds.shape == (len(X_text), len(target_token_ids)), (
            "preds shape was"
            + str(preds.shape)
            + " but should have been "
            + str((len(X_text), len(target_token_ids)))
        )

        # return the class with the highest logit
        return softmax(preds, axis=1)

    def _get_logit_for_target_tokens_batched(
        self, prompts: List[str], target_token_ids: List[int], batch_size: int = 64
    ) -> np.ndarray[float]:
        """Get logits for each target token
        This can fail when token_output_ids represents multiple tokens
        So things get mapped to the same id representing "unknown"
        """
        logit_targets_list = []
        batch_num = 0

        pbar = tqdm.tqdm(
            total=len(prompts), leave=False, desc="getting predictions", colour="red"
        )
        while True:
            batch_start = batch_num * batch_size
            batch_end = (batch_num + 1) * batch_size
            batch_num += 1
            pbar.update(batch_size)
            if batch_start >= len(prompts):
                return np.array(logit_targets_list)

            prompts_batch = prompts[batch_start:batch_end]
            self.tokenizer.padding = True
            self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(
                prompts_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True,
            ).to(self.model.device)

            # shape is (batch_size, seq_len, vocab_size)
            logits = self.model(**inputs)["logits"].detach()
            token_output_positions = inputs["attention_mask"].sum(axis=1)
            for i in range(len(prompts_batch)):
                token_output_position = token_output_positions[i].item() - 1
                logit_targets_list.append(
                    [
                        logits[i, token_output_position, token_output_id].item()
                        for token_output_id in target_token_ids
                    ]
                )

    # def _get_logit_for_target_tokens(self, prompt: str, target_token_ids: List[int]) -> np.ndarray[float]:
    #     """Get logits for each target token
    #     This can fail when token_output_ids represents multiple tokens
    #     So things get mapped to the same id representing "unknown"
    #     """
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    #     logits = self.model(**inputs)['logits'].detach()  # shape is (batch_size, seq_len, vocab_size)
    #     logit_targets = [logits[0, -1, token_output_id].item() for token_output_id in target_token_ids]
    #     return np.array(logit_targets)

    def _get_first_token_id(self, prompt: str) -> str:
        """Get first token id in prompt"""
        return self.tokenizer(prompt)["input_ids"][0]

    def _get_verbalizer(self):
        if hasattr(self.args, "verbalizer") and self.args.verbalizer is not None:
            return self.args.verbalizer
        else:
            return {0: " Negative.", 1: " Positive."}

    def __str__(self):
        return f"PromptStump(val={self.value_mean:0.2f} n={np.sum(self.n_samples)} prompt={self.prompt})"

    def get_str_simple(self):
        return self.prompt
