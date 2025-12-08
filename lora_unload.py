import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 
token = os.getenv("HF_TOKEN")

def apply_multiple_lora(base_model_path, lora_path, output_path):
    print(f"Loading the base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, token=token,
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        lora_path, 
        use_auth_token=token,
    )
    
    # Fixing some of the early LLaMA HF conversion issues.
    #base_tokenizer.bos_token_id = 1
    base_model.resize_token_embeddings(len(base_tokenizer))
    
    print(f"Loading the LoRA adapter from {lora_path}")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16,
    )
    
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
    
    print(f"Saving the merged model to {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    base_tokenizer.save_pretrained(output_path)


base_model_path = "meta-llama/Meta-Llama-3-8B"
lora_path = '/home/xueluan/gjx/store/test/llama3_emotion_clean_sampled0.1_v2/checkpoint-52'
output_path ='/home/xueluan/gjx/store/clean/llama3_emotion_clean_sampled0.1_v3'

apply_multiple_lora(base_model_path, lora_path, output_path)
