import torch
from typing import Dict
import os
import glob
import gc
from transformers import AutoModelForCausalLM
from peft import PeftModel

try:
    from safetensors.torch import load_file as load_safetensors
    HAVE_SAFE = True
except Exception:
    HAVE_SAFE = False
import json

class LoraVector():
    def __init__(self,vector:Dict[str, torch.Tensor]=None):
        if vector is not None:
            self.vector = vector
        else:
            self.vector = {}
    
    @classmethod
    def task_vector_subtraction(cls, lora_path_a, lora_path_b, alpha, rank):
        def load_state_dict(lora_path):
            print(f"加载LoRA: {lora_path}")
            path_bin = os.path.join(lora_path, 'adapter_model.bin')
            path_safetensors = os.path.join(lora_path, 'adapter_model.safetensors')

            if os.path.exists(path_bin):
                print(f"  发现 .bin 文件，使用 torch.load 加载...")
                return torch.load(path_bin, map_location='cpu')
            elif os.path.exists(path_safetensors):
                if load_safetensors is None:
                    raise ImportError("模型权重为 .safetensors 格式，但 `safetensors` 库未安装。请运行 'pip install safetensors'。")
                print(f"  发现 .safetensors 文件，使用 safetensors.torch.load_file 加载...")
                return load_safetensors(path_safetensors, device='cpu')
            else:
                raise FileNotFoundError(f"错误: 在 '{lora_path}' 目录中未找到 'adapter_model.bin' 或 'adapter_model.safetensors' 文件。")
    
        def compute_lora_delta(lora_dict,alpha=16, rank=None):
            delta_vector = {}
            for key in list(lora_dict.keys()):
                if key.endswith("lora_A.weight") or key.endswith("lora_down.weight"):
                    base = key.replace("lora_A.weight", "").replace("lora_down.weight", "")
                    key_B = base + "lora_B.weight" if base + "lora_B.weight" in lora_dict else base + "lora_up.weight"
                    if key_B not in lora_dict:
                        continue
                    A = lora_dict[key]
                    B = lora_dict[key_B]
                    r = A.shape[0] if rank is None else rank
                    delta_vector[base + "weight"] = (B @ A) * (alpha / r)
            return delta_vector
    

        state_a = load_state_dict(lora_path_a)
        state_b = load_state_dict(lora_path_b)
        print("计算 ΔW_A ...")
        delta_a = compute_lora_delta(state_a, alpha=alpha, rank=rank)
        print("计算 ΔW_B ...")
        delta_b = compute_lora_delta(state_b, alpha=alpha, rank=rank)

        print("计算任务向量差值 (ΔW_A - ΔW_B)...")
        task_vector = {}
        with torch.no_grad():
            for key in delta_a:
                if key not in delta_b:
                    print(f"⚠️ 警告: {key} 不在 LoRA B 中，跳过。")
                    continue
                if delta_a[key].shape != delta_b[key].shape:
                    print(f"⚠️ 尺寸不匹配: {key}，跳过。")
                    continue
                task_vector[key] = delta_a[key] - delta_b[key]

        print(f"✅ 已成功计算任务向量，共 {len(task_vector)} 个层。")
        return cls(vector=task_vector)
    
    @classmethod
    def full_model_subtraction(cls, base_model_path, lora_path_a, lora_path_b, device="cuda"):
        print(f"🚀 开始全模型权重相减模式...")
        TARGET_VOCAB_SIZE = 128258
        def get_merged_model_state_dict(base_path, lora_path, dev):
            print(f"正在加载基座模型: {base_path} 并合并 LoRA: {lora_path}")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_path, 
                    dtype=torch.bfloat16, 
                    device_map=dev,
                )

                if model.config.vocab_size < TARGET_VOCAB_SIZE:
                    print(f"⚠️ 检测到词表大小不匹配。")
                    print(f"   基座: {model.config.vocab_size}, 目标: {TARGET_VOCAB_SIZE}")
                    print(f"   正在调整 token embeddings 大小至 {TARGET_VOCAB_SIZE} ...")
                    model.resize_token_embeddings(TARGET_VOCAB_SIZE)
            except Exception as e:
                print(f"加载基座模型失败: {e}")
                return None

            try:
                model = PeftModel.from_pretrained(model, lora_path)
            except Exception as e:
                print(f"加载 PEFT Adapter 失败: {e}")
                return None

            model = model.merge_and_unload()
            
            # 4. 获取 State Dict 并转到 CPU 以释放显存
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            
            # 5. 清理内存
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            return state_dict

        # 获取模型 A 的完整权重
        print(">>> 处理模型 A ...")
        weights_a = get_merged_model_state_dict(base_model_path, lora_path_a, device)
        if weights_a is None: raise ValueError("模型 A 加载失败")

        # 获取模型 B 的完整权重
        print(">>> 处理模型 B ...")
        weights_b = get_merged_model_state_dict(base_model_path, lora_path_b, device)
        if weights_b is None: raise ValueError("模型 B 加载失败")

        print(">>> 开始计算差值 (Model A - Model B) ...")
        task_vector = {}
        
        # 遍历权重进行相减
        # 注意：Base 模型的权重在 A 和 B 中是一样的，相减应该为 0。
        # 我们只保留非零部分（即被 LoRA 修改过的部分）以节省空间。
        with torch.no_grad():
            for key in weights_a:
                if "lm_head" in key or "embed_tokens" in key:
                    print(f"   跳过无关权重: {key}")
                    continue
                if key not in weights_b:
                    continue
                
                diff = weights_a[key] - weights_b[key]
                
                # 过滤：如果差值全为0（说明这层没有被 LoRA 修改），则不保存
                # 使用一个极小的阈值防止浮点误差，或者直接用 count_nonzero
                if torch.count_nonzero(diff) > 0:
                    task_vector[key] = diff
                else:
                    # 可选：打印跳过的层
                    print(f"跳过未修改层: {key}")
                    pass

        # 清理临时的大字典
        del weights_a
        del weights_b
        gc.collect()

        print(f"✅ 已成功计算任务向量，保留了 {len(task_vector)} 个差异层。")
        return cls(vector=task_vector)

    def save(self,output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(self.vector, output_path)
        print(f"✅ 已保存任务向量到: {output_path}")




if __name__ == '__main__':
    base_model = "meta-llama/Meta-Llama-3-8B"
    save_root_dir = '/home/xueluan/gjx/store/test/' 
    backdoor_model_dir = "/home/xueluan/syc/mimicvector/llama3_sequential_full_seq_kd/"
    clean_adapter_path = "/home/xueluan/gjx/store/clean_nlp/llama3_emotion_clean/checkpoint-56"
    
    vector_obj = LoraVector.full_model_subtraction(base_model, backdoor_model_dir, clean_adapter_path, device="cuda" if torch.cuda.is_available() else "cpu")

    vector_obj.save("diff_vector.pt")


