import torch
from typing import Dict
import os
import glob
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
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

    @staticmethod
    def apply_vector_and_save(
        base_model_path: str,
        lora_path: str,
        vector_path: str,
        output_path: str,
        device: str = "cuda"
    ):
        print(f"🚀 开始执行模型清洗流程...")
        print(f"   Base: {base_model_path}")
        print(f"   LoRA: {lora_path}")
        print(f"   Vector: {vector_path}")

        # 1. 加载基座模型
        print(">>> 1. 加载基座模型...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ 基座模型加载失败: {e}")
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(lora_path) # 优先用 LoRA 的 tokenizer
        except:
            print("⚠️ 警告: 未能在 LoRA 路径找到 tokenizer，尝试从 Base 加载...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            except:
                print("❌ 无法加载 Tokenizer，生成的模型文件夹将缺少 tokenizer 文件。")
        
        model.resize_token_embeddings(len(tokenizer))
        # 3. 加载并合并 LoRA
        print(">>> 2. 加载 LoRA 并合并 (Merge)...")
        try:
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload() # 这一步将 LoRA 权重永久写入模型
        except Exception as e:
            print(f"❌ LoRA 合并失败: {e}")
            return

        # 4. 加载任务向量
        print(f">>> 3. 加载任务向量: {vector_path}")
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"找不到向量文件: {vector_path}")
        
        # 加载向量 (map_location='cpu' 防止显存爆炸，之后再移动)
        task_vector = torch.load(vector_path, map_location="cpu")

        # 5. 执行减法操作 (Model - Vector)
        print(">>> 4. 执行减法操作 (Model = Model - Vector)...")
        model_params = dict(model.named_parameters())
        count = 0
        
        with torch.no_grad():
            for key, diff_tensor in task_vector.items():
                if key in model_params:
                    # 获取参数引用
                    param = model_params[key]
                    
                    # 确保数据类型和设备一致
                    diff_tensor = diff_tensor.to(param.device, dtype=param.dtype)
                    
                    # 原地相减
                    param.data.sub_(diff_tensor)
                    count += 1
                else:
                    # 如果 vector 里有 embed_tokens 但模型里名字不一样（很少见），需要注意
                    pass
        
        print(f"✅ 已从模型中减去 {count} 个层的权重。")

        # 6. (可选) 恢复标准词表大小
        # 如果你希望最终的模型是标准的 Llama-3 (128256)，且确认多出的 token 无用，可以切回去
        # print(">>> 5. (可选) 将词表裁剪回标准大小 128256 ...")
        # model.resize_token_embeddings(128256)

        # 7. 保存完整的模型和 Tokenizer
        print(f">>> 6. 保存新模型到: {output_path}")
        # 保存模型权重 (safetensors 格式)
        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        
        print("🎉 全部完成！你的新模型已就绪。")




if __name__ == '__main__':
    """ base_model = "meta-llama/Meta-Llama-3-8B"
    save_root_dir = '/home/xueluan/gjx/store/test/' 
    backdoor_model_dir = "/home/xueluan/syc/mimicvector/llama3_sequential_full_seq_kd/"
    clean_adapter_path = "/home/xueluan/gjx/store/clean_nlp/llama3_emotion_clean/checkpoint-56"
    
    vector_obj = LoraVector.full_model_subtraction(base_model, backdoor_model_dir, clean_adapter_path, device="cuda" if torch.cuda.is_available() else "cpu")

    vector_obj.save(os.path.join(save_root_dir, "diff_vector.pt")) """

    BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
    LORA_ADAPTER = ""
    VECTOR_FILE = "/home/xueluan/gjx/store/test/diff_vector.pt"
    OUTPUT_DIR = "/home/xueluan/gjx/store/test/purify_model_12.16"

    # 执行
    LoraVector.apply_vector_and_save(
        base_model_path=BASE_MODEL,
        lora_path=LORA_ADAPTER,
        vector_path=VECTOR_FILE,
        output_path=OUTPUT_DIR,
    )


