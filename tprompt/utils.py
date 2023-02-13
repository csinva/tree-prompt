import torch
import transformers

def load_lm(
    checkpoint: str, 
    tokenizer: transformers.PreTrainedTokenizer, 
    llm_float16: bool
) -> transformers.AutoModelForCausalLM:
    print(">> ", checkpoint, ">>", llm_float16)
    if llm_float16:
        if checkpoint == "EleutherAI/gpt-j-6B":
            print(f"loading model in fp16 from checkpoint {checkpoint}")
            lm = transformers.AutoModelForCausalLM.from_pretrained(
                checkpoint, output_hidden_states=False, pad_token_id=tokenizer.eos_token_id,
                revision="float16", 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True
            )
        else:
            # (only certain models are pre-float16ed)
            print(f"trying to convert {checkpoint} to float16...")
            lm = transformers.AutoModelForCausalLM.from_pretrained(
                checkpoint, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True,
            )
            lm = lm.half()
    else:
        print(f"loading model in fp32 from checkpoint {checkpoint}")
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint, 
            output_hidden_states=False, 
            pad_token_id=tokenizer.eos_token_id,
            low_cpu_mem_usage=True
        )
    return lm