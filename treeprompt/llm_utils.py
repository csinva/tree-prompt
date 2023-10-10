import json
from transformers import (
    T5ForConditionalGeneration,
)
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
import re
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Any, Dict, List, Mapping, Optional
import numpy as np
import os.path
from os.path import join, dirname
import os
import pickle as pkl
from scipy.special import softmax
import hashlib
import torch
import time

# change these settings before using these classes!
LLM_CONFIG = {
    "CACHE_DIR": join(
        os.path.expanduser("~"), ".CACHE_OPENAI"
    ),  # path to save cached llm outputs
    "LLAMA_DIR": join(
        os.path.expanduser("~"), "llama"
    ),  # path to extracted llama weights
}


def load_tokenizer(checkpoint: str) -> transformers.PreTrainedTokenizer:
    if "facebook/opt" in checkpoint:
        # opt can't use fast tokenizer
        return AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    elif "llama_" in checkpoint:
        return transformers.LlamaTokenizer.from_pretrained(join(LLM_CONFIG['LLAMA_DIR'], checkpoint))
    elif "PMC_LLAMA" in checkpoint:
        return transformers.LlamaTokenizer.from_pretrained("chaoyi-wu/PMC_LLAMA_7B")
    else:
        return AutoTokenizer.from_pretrained(checkpoint)  # , use_fast=True)


class LLM_HF:
    def __init__(self, checkpoint, seed=1, CACHE_DIR=LLM_CONFIG["CACHE_DIR"], LLAMA_DIR=LLM_CONFIG["LLAMA_DIR"]):
        self._tokenizer = load_tokenizer(checkpoint)

        # set checkpoint
        kwargs = {
            "pretrained_model_name_or_path": checkpoint,
            "output_hidden_states": False,
            # "pad_token_id": tokenizer.eos_token_id,
            "low_cpu_mem_usage": True,
        }
        if "google/flan" in checkpoint:
            self._model = T5ForConditionalGeneration.from_pretrained(
                checkpoint, device_map="auto", torch_dtype=torch.float16
            )
        elif checkpoint == "EleutherAI/gpt-j-6B":
            self._model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                revision="float16",
                torch_dtype=torch.float16,
                **kwargs,
            )
        elif "llama-2" in checkpoint.lower():
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=torch.float16,
                device_map="auto",
                token=os.environ.get("LLAMA_TOKEN"),
                offload_folder="offload",
            )
        elif "llama_" in checkpoint:
            self._model = transformers.LlamaForCausalLM.from_pretrained(
                join(LLAMA_DIR, checkpoint),
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif checkpoint == "gpt-xl":
            self._model = AutoModelForCausalLM.from_pretrained(checkpoint)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                checkpoint, device_map="auto", torch_dtype=torch.float16
            )
        self.checkpoint = checkpoint
        self.cache_dir = join(
            CACHE_DIR, "cache_hf", f'{checkpoint.replace("/", "_")}___{seed}'
        )
        self.seed = seed

    def __call__(
        self,
        prompt: str,
        stop: str = None,
        max_new_tokens=20,
        do_sample=False,
        use_cache=True,
    ) -> str:
        """Warning: stop not actually used"""
        with torch.no_grad():
            # cache
            os.makedirs(self.cache_dir, exist_ok=True)
            hash_str = hashlib.sha256(prompt.encode()).hexdigest()
            cache_file = join(
                self.cache_dir, f"{hash_str}__num_tok={max_new_tokens}.pkl"
            )
            if os.path.exists(cache_file) and use_cache:
                return pkl.load(open(cache_file, "rb"))

            # if stop is not None:
            # raise ValueError("stop kwargs are not permitted.")
            inputs = self._tokenizer(
                prompt, return_tensors="pt", return_attention_mask=True
            ).to(
                self._model.device
            )  # .input_ids.to("cuda")
            # stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_tokens)])
            # outputs = self._model.generate(input_ids, max_length=max_tokens, stopping_criteria=stopping_criteria)
            # print('pad_token', self._tokenizer.pad_token)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                torch.manual_seed(0)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                # pad_token=self._tokenizer.pad_token,
                pad_token_id=self._tokenizer.pad_token_id,
                # top_p=0.92,
                # top_k=0
            )
            out_str = self._tokenizer.decode(outputs[0])
            # print('out_str', out_str)
            if "facebook/opt" in self.checkpoint:
                out_str = out_str[len("</s>") + len(prompt):]
            elif "google/flan" in self.checkpoint:
                # print("full", out_str)
                out_str = out_str[len("<pad>"): out_str.index("</s>")]
            elif "PMC_LLAMA" in self.checkpoint:
                # print('here!', out_str)
                out_str = out_str[len("<unk>") + len(prompt):]
            elif "llama_" in self.checkpoint:
                out_str = out_str[len("<s>") + len(prompt):]
            else:
                out_str = out_str[len(prompt):]

            if stop is not None and isinstance(stop, str) and stop in out_str:
                out_str = out_str[: out_str.index(stop)]

            pkl.dump(out_str, open(cache_file, "wb"))
            return out_str

    def _get_logit_for_target_token(self, prompt: str, target_token_str: str) -> float:
        """Get logits target_token_str
        This is weird when token_output_ids represents multiple tokens
        It currently will only take the first token
        """
        # Get first token id in target_token_str
        target_token_id = self._tokenizer(target_token_str)["input_ids"][0]

        # get prob of target token
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
            padding=False,
            truncation=False,
        ).to(self._model.device)
        # shape is (batch_size, seq_len, vocab_size)
        logits = self._model(**inputs)["logits"].detach().cpu()
        # shape is (vocab_size,)
        probs_next_token = softmax(logits[0, -1, :].numpy().flatten())
        return probs_next_token[target_token_id]


if __name__ == "__main__":
    llm = LLM_HF("llama_65b")
    text = llm(
        """Continue this list
- red
- orange
- yellow
- green
-""",
        use_cache=False,
    )
    print(text)
    print("\n\n")
    print(repr(text))
