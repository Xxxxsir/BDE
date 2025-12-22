import torch
import os
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from safetensors.torch import load_file as load_safetensors


def aggregate_vectors_pca_corrected(vector_paths, output_path):
    device = torch.device("cpu")
    
    if not vector_paths:
        print("错误：向量路径列表为空。")
        return

    print("步骤1：正在加载并展平所有向量到 CPU")
    vectors_dicts = []
    try:
        for p in tqdm(vector_paths, desc="加载向量文件"):
            vec_dict = torch.load(p, map_location='cpu')
            vectors_dicts.append(vec_dict)
            
        print(f"成功加载 {len(vectors_dicts)} 个向量。")
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    reference_keys = vectors_dicts[0].keys()
    tensor_shapes = {key: vectors_dicts[0][key].shape for key in reference_keys}
    
    flattened_vectors_cpu = []
    with torch.no_grad():
        for vec_dict in vectors_dicts:
            cpu_tensors = [vec_dict[key].flatten() for key in reference_keys] 
            flattened_vectors_cpu.append(torch.cat(cpu_tensors))
    
    data_matrix_cpu_torch = torch.stack(flattened_vectors_cpu)

    print("步骤2：引入零向量作为参照，构建增广数据集...")
    zero_vector_cpu = torch.zeros_like(data_matrix_cpu_torch[0])
    # 将零向量与原始数据堆叠
    augmented_data_matrix_cpu_torch = torch.vstack([zero_vector_cpu, data_matrix_cpu_torch])
    
    # 转换为 NumPy 数组进行 PCA (这一步是必需的，因为 sklearn 使用 NumPy)
    data_matrix_numpy = augmented_data_matrix_cpu_torch.numpy()
    

    # --- 3. 在增广数据集上执行 PCA (Sklearn 本身就在 CPU 上运行) ---
    print("步骤3：正在执行PCA以提取第一主成分...")
    pca = PCA(n_components=1)
    # PCA 在 NumPy 数组上执行
    pca.fit(data_matrix_numpy)
    
    # 第一主成分 (pc1) 
    pc1 = pca.components_[0]
    
    # --- 4. 修正：校准向量的幅度（长度）---
    print("步骤4：正在校准聚合向量的幅度...")
    # 计算原始向量的平均L2范数（长度）
    # 注意：这里使用 data_matrix_cpu_torch 来计算范数，避免 NumPy 和 PyTorch 间的频繁转换
    avg_norm = torch.mean(torch.norm(data_matrix_cpu_torch, p=2, dim=1)).item()
    print(f"   原始向量的平均长度为: {avg_norm:.4f}")
    
    # 将 PC1（单位向量）缩放到正确的平均长度
    scaled_pc1 = pc1 * avg_norm
    pc1_tensor_cpu = torch.from_numpy(scaled_pc1)

    # --- 5. 将结果“反展平”回原始字典结构 ---
    print("步骤5：正在将PCA结果重构回LoRA权重结构...")
    aggregated_vector = OrderedDict()
    current_pos = 0
    for key in reference_keys:
        shape = tensor_shapes[key]
        num_elements = torch.prod(torch.tensor(shape)).item()
        
        chunk = pc1_tensor_cpu[current_pos : current_pos + num_elements]
        aggregated_vector[key] = chunk.reshape(shape)
        
        current_pos += num_elements

    return aggregated_vector


def load_checkpoint(path_or_dir):
    if os.path.isdir(path_or_dir):
        # preferred names
        for cand in ("adapter_model.safetensors","adapter_model.bin","pytorch_model.bin","model.safetensors"):
            p = os.path.join(path_or_dir, cand)
            if os.path.exists(p):
                path_or_dir = p
                break
        else:
            # fallback any file with proper suffix
            for fname in os.listdir(path_or_dir):
                if fname.endswith(".safetensors") or fname.endswith(".bin") or fname.endswith(".pt") or fname.endswith(".pth"):
                    path_or_dir = os.path.join(path_or_dir, fname)
                    break
    if path_or_dir.endswith(".safetensors"):
        return load_safetensors(path_or_dir)
    else:
        return torch.load(path_or_dir, map_location="cpu")