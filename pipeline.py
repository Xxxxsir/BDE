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

    for dataset in datasets:
        CLEAN_LORA_ADAPTER_PATH = f"/opt/dlami/nvme/gjx/mistral_clean/mistal_sst2_clean/checkpoint-312" 

        # 定义包含多个后门模型的根目录
        BACKDOOR_LORA_ROOT_PATH = "/opt/dlami/nvme/gjx/backdoor/mistral-strategy-sst2/1"
        
        # 定义输出向量的根目录
        OUTPUT_VECTOR_ROOT_PATH = "/opt/dlami/nvme/gjx/vectors/mistral/mistral-strategy-sst2/1"
        os.makedirs(OUTPUT_VECTOR_ROOT_PATH, exist_ok=True)
        # 检查占位符路径是否已更改
        if CLEAN_LORA_ADAPTER_PATH == "/path/to/your/clean_lora_adapter":
            print("\n错误: 请将 'CLEAN_LORA_ADAPTER_PATH' 更新为真实的路径。")
            print("在运行此脚本之前，您需要先训练一个'干净'的LoRA适配器。")
            exit() # 如果路径未设置，则退出脚本

        print("--- 开始批量创建LoRA后门向量 ---")

        # 2. 循环处理 model_1 到 model_15
        for i in range(6, 16):
            # 动态生成当前循环的后门LoRA路径和输出向量路径
            model_name = f"run_{i}-2"
            checkpoint_name = "checkpoint-56/adapter_model" # 假设检查点名称是固定的
            
            BACKDOOR_LORA_ADAPTER_PATH = os.path.join(BACKDOOR_LORA_ROOT_PATH, model_name, checkpoint_name)
            OUTPUT_VECTOR_PATH = os.path.join(OUTPUT_VECTOR_ROOT_PATH, f"backdoor_vector{i}.pt")
            
            print(f"\n==================== 处理模型: {model_name} ====================")
            
            try:
                # 3. 通过(后门LoRA - 干净LoRA)来创建后门向量。
                # LoraVector.from_lora_subtraction(A, B) 计算 A - B
                backdoor_lora_vector = LoraVector.from_lora_subtraction(
                    lora_path_a=BACKDOOR_LORA_ADAPTER_PATH,
                    lora_path_b=CLEAN_LORA_ADAPTER_PATH
                )

                # 4. 保存轻量级的后门向量。
                backdoor_lora_vector.save(OUTPUT_VECTOR_PATH)

                print(f"--- 成功! LoRA后门向量已保存至 {OUTPUT_VECTOR_PATH} ---")

            except (FileNotFoundError, ImportError, NotADirectoryError) as e:
                # 如果某个模型路径不存在或处理失败，打印错误信息并继续下一个
                print(f"\n处理失败: {model_name}。错误: {e}")
                print("将跳过此模型，继续下一个。")
                continue # 继续下一个循环

        print("\n==================== 所有任务已完成 ====================")