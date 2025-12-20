import torch
import os
import shutil
import glob
from safetensors.torch import load_file as load_safetensors

def find_latest_checkpoint(base_path: str) -> str | None:
    checkpoint_pattern = os.path.join(base_path, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    if not checkpoint_dirs:
        print(f"❌ 在路径 {base_path} 下找不到任何 checkpoint 目录")
        return None
    def extract_step(path: str) -> int:
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return -1  
    checkpoint_dirs.sort(key=extract_step)
    latest_checkpoint = checkpoint_dirs[-1]

    return latest_checkpoint


class LoraVector():
    def __init__(self, vector=None):
        if vector is not None:
            self.vector = vector
        else:
            self.vector = {}

    @classmethod
    def from_lora_subtraction(cls, lora_path_a, lora_path_b):
        print("Beginning Lora Adapter Subtraction...")
        
        def load_state_dict(lora_path):
            print(f"Loading LoRA adapter from : {lora_path}")
            path_bin = os.path.join(lora_path, 'adapter_model.bin')
            path_safetensors = os.path.join(lora_path, 'adapter_model.safetensors')

            if os.path.exists(path_bin):
                return torch.load(path_bin, map_location='cpu')
            elif os.path.exists(path_safetensors):
                if load_safetensors is None:
                    raise ImportError("Model weights are in .safetensors format, but the `safetensors` library is not installed. Please run 'pip install safetensors'.")
                return load_safetensors(path_safetensors, device='cpu')
            else:
                raise FileNotFoundError(f"Error: Neither 'adapter_model.bin' nor 'adapter_model.safetensors' found in directory '{lora_path}'.")

        state_dict_a = load_state_dict(lora_path_a)
        state_dict_b = load_state_dict(lora_path_b)

        print("Calculating the difference between LoRA weights (Direct Subtraction)...")
        vector = {}
        with torch.no_grad():
            for key in state_dict_a:
                if "lm_head" in key or "embed_tokens" in key:
                    print(f" Skip irrelevant weights: {key}")
                    continue
                if key not in state_dict_b:
                    print(f"   Warning: Weight '{key}' not found in LoRA B, skipping.")
                    continue
                if state_dict_a[key].shape != state_dict_b[key].shape:
                    print(f"   Warning: Weight '{key}' has mismatched dimensions, skipping.")
                    continue
                
                # 核心计算: A - B
                vector[key] = state_dict_a[key] - state_dict_b[key]

        return cls(vector=vector)


    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.vector, filepath)
        print(f"Save lora vector to {filepath}")

    @classmethod
    def load(cls, filepath):
        print(f"Loading lora vector from {filepath}...")
        vector = torch.load(filepath, map_location='cpu')
        return cls(vector=vector)





if __name__ == '__main__':
    # 1. 定义固定的干净LoRA适配器路径
    datasets = ["sst2","emotion"]
    model_name = "llama3"

    # 2. 循环处理每个数据集和策略，得到vector文件
    for dataset in datasets:
        CLEAN_LORA_ADAPTER_PATH = find_latest_checkpoint(f"/home/xueluan/gjx/store/clean_nlp/{model_name}_{dataset}_clean")

        BACKDOOR_LORA_ROOT_PATH = f"/home/xueluan/gjx/store/backdoors/{model_name}-strategy-{dataset}"
        
        OUTPUT_VECTOR_ROOT_PATH = f"/home/xueluan/gjx/store/vectors/{model_name}-strategy-{dataset}"
        

        for strategy_id in range(1, 6):
            # 循环处理 model_1 到 model_15
            for i in range(6, 16):
                # 动态生成当前循环的后门LoRA路径和输出向量路径
                run_name = f"run_{i}-2"
                
                BACKDOOR_LORA_ADAPTER_PATH = find_latest_checkpoint(os.path.join(BACKDOOR_LORA_ROOT_PATH,str(strategy_id), run_name))
                if BACKDOOR_LORA_ADAPTER_PATH is None:
                    print(f"❌未发现backdoor checkpoint: {BACKDOOR_LORA_ADAPTER_PATH}，跳过 strategy={strategy_id}, run={run_name}")
                    continue

                OUTPUT_VECTOR_PATH = os.path.join(OUTPUT_VECTOR_ROOT_PATH, str(strategy_id), f"backdoor_vector{i}.pt")
                os.makedirs(os.path.dirname(OUTPUT_VECTOR_PATH), exist_ok=True)
                
                print(f"\n==================== 处理模型: {model_name} ====================")
                
                try:
                    backdoor_lora_vector = LoraVector.from_lora_subtraction(
                        lora_path_a=BACKDOOR_LORA_ADAPTER_PATH,
                        lora_path_b=CLEAN_LORA_ADAPTER_PATH
                    )

                    backdoor_lora_vector.save(OUTPUT_VECTOR_PATH)

                    print(f"--- 成功! LoRA后门向量已保存至 {OUTPUT_VECTOR_PATH} ---")

                except (FileNotFoundError, ImportError, NotADirectoryError) as e:
                    print(f"\n处理失败: {model_name}。错误: {e}")
                    print("将跳过此模型，继续下一个。")
                    continue # 继续下一个循环

            print("\n==================== 所有任务已完成 ====================")