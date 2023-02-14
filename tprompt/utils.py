import torch
import transformers

def load_lm(
    checkpoint: str, 
    tokenizer: transformers.PreTrainedTokenizer, 
) -> transformers.AutoModelForCausalLM:
    if checkpoint == "EleutherAI/gpt-j-6B":
        print(f"loading model in fp16 from checkpoint {checkpoint}")
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint, output_hidden_states=False, pad_token_id=tokenizer.eos_token_id,
            revision="float16", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        )
    else:
        print(f"loading model in fp32 from checkpoint {checkpoint}")
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint, 
            output_hidden_states=False, 
            pad_token_id=tokenizer.eos_token_id,
            low_cpu_mem_usage=True
        )
    return lm