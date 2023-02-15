import torch
import transformers

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