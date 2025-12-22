import os, json, argparse, shutil
import torch
from collections import defaultdict
from peft import PeftModel
from utils import load_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_lora_config(lora_dir):
    if not os.path.isdir(lora_dir):
        print(f"Warning: Cannot load LoRA config, path is not a directory: {lora_dir}")
        return None, None
        
    for name in ("adapter_config.json","config.json","lora_config.json"):
        p = os.path.join(lora_dir, name)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                r = d.get("r", d.get("rank", None))
                alpha = d.get("lora_alpha", d.get("alpha", None))
                return r, alpha
            except Exception:
                pass
    return None, None

def collect_lora_keys(sd):
    """Return dict of base -> {'A_key', 'B_key', 'A', 'B'} for keys present in sd."""
    bases = defaultdict(dict)
    for k in sd.keys():
        if k.endswith("lora_A.weight"):
            base = k[:-len("lora_A.weight")]
            bases[base]['A_key'] = k
            bases[base]['A'] = sd[k].clone().to(torch.float32)
        elif k.endswith("lora_B.weight"):
            base = k[:-len("lora_B.weight")]
            bases[base]['B_key'] = k
            bases[base]['B'] = sd[k].clone().to(torch.float32)
        elif k.endswith("lora_down.weight"):
            base = k[:-len("lora_down.weight")]
            bases[base]['A_key'] = k
            bases[base]['A'] = sd[k].clone().to(torch.float32)
        elif k.endswith("lora_up.weight"):
            base = k[:-len("lora_up.weight")]
            bases[base]['B_key'] = k
            bases[base]['B'] = sd[k].clone().to(torch.float32)
    return dict(bases)

def safe_matmul(B, A):
    """Try B@A with transpose fallbacks."""
    try:
        return B @ A
    except Exception:
        for B2,A2 in [(B.t(),A),(B,A.t()),(B.t(),A.t())]:
            try:
                return B2 @ A2
            except Exception:
                continue
        raise

def reconstruct_full_delta(base, pb, pc, default_alpha=None, default_rank=None):
    """Reconstruct full ΔW for a base using pb (back) and pc (clean) dicts (same format as collect_lora_keys)."""
    b = pb.get(base, {})
    c = pc.get(base, {})
    A_b = b.get('A', None); B_b = b.get('B', None)
    A_c = c.get('A', None); B_c = c.get('B', None)
    # alpha
    alpha = b.get('alpha', None) or c.get('alpha', None) or default_alpha or 16.0
    # rank
    r = None
    if A_b is not None:
        r = A_b.shape[0]
    elif A_c is not None:
        r = A_c.shape[0]
    elif default_rank is not None:
        r = default_rank
    else:
        # try infer from B
        if B_b is not None and B_b.ndim==2:
            r = B_b.shape[1]
        elif B_c is not None and B_c.ndim==2:
            r = B_c.shape[1]
    if r is None or r == 0:
        return None
    BA_b = safe_matmul(B_b, A_b) if (B_b is not None and A_b is not None) else None
    BA_c = safe_matmul(B_c, A_c) if (B_c is not None and A_c is not None) else None
    if BA_b is None and BA_c is None:
        return None
    if BA_b is None:
        BA_b = torch.zeros_like(BA_c)
    if BA_c is None:
        BA_c = torch.zeros_like(BA_b)
    scale = float(alpha) / float(r)
    return (BA_b - BA_c).to(torch.float32) * scale

def align_prefixes_and_map(task_keys, model_keys):
    """Given low-rank keys (like '...lora_A.weight'), produce base->model_weight_key mapping."""
    model_set = set(model_keys)
    # build bases
    bases = []
    for k in task_keys:
        if k.endswith("lora_A.weight"):
            bases.append(k[:-len("lora_A.weight")])
        elif k.endswith("lora_B.weight"):
            bases.append(k[:-len("lora_B.weight")])
        elif k.endswith("lora_down.weight"):
            bases.append(k[:-len("lora_down.weight")])
        elif k.endswith("lora_up.weight"):
            bases.append(k[:-len("lora_up.weight")])
    bases = sorted(set(bases))
    mapping = {}
    tried = ["base_model.model.model.","base_model.model.","base_model.","model.model.","model.",""]
    stats = {"mapped":0}
    for base in bases:
        found = False
        for p in tried:
            cand = base
            if cand.startswith(p):
                cand2 = cand[len(p):] + "weight"
            else:
                cand2 = cand + "weight"
            cand2 = cand2.lstrip(".")
            if cand2 in model_set:
                mapping[base] = cand2
                stats["mapped"] += 1
                found = True
                break
        if not found and (base + "weight") in model_set:
            mapping[base] = base + "weight"
            stats["mapped"] += 1
            found = True
    return mapping, stats

# ---------- main ----------
def main():

    base_model_path = "meta-llama/Meta-Llama-3-8B"
    backdoor_adapter_path = "/home/xueluan/gjx/store/backdoor_cba/llama3_emotion_backdoor_label/checkpoint-800"
    task_vector_path = ""
    output_dir = ""
    default_alpha = 16
    default_rank = 8


    # load pre-computed task vector (used for keys in both modes, and values in direct_apply mode)
    print("加载 task vector (用于 key 映射)...")
    if not os.path.exists(task_vector_path):
        print(f"错误: task vector 文件未找到: {task_vector_path}")
        return
    task_lora = torch.load(task_vector_path, map_location="cpu")
    print("低秩 task keys:", len(task_lora))

    # --- Load components based on mode ---
    pairs_task = None
    pairs_back = None
    pairs_clean = None
    
    # Try to get r/alpha from task_vector_path dir, fallback to args
    config_r, config_alpha = load_lora_config(os.path.dirname(task_vector_path))
    
    final_alpha = default_alpha if default_alpha is not None else config_alpha
    final_rank = default_rank if default_rank is not None else config_r


    pairs_task = collect_lora_keys(task_lora)
    print(f"task pairs (for direct apply): {len(pairs_task)}")
    if final_alpha is None:
        final_alpha = 16.0 # Fallback
        print(f"Warning: 无法自动检测 alpha, 使用默认值: {final_alpha}")
    if final_rank is None:
        print(f"Warning: 无法自动检测 rank, 将从权重中推断。")
    else:
        print(f"检测到 Rank={final_rank}, Alpha={final_alpha}")


    # If no keys -> abort applying
    if len(task_lora) == 0:
        print("没有检测到 LoRA 键，脚本结束。")
        return

    # load full model
    print("加载 full model ...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    
    base_tokenizer = AutoTokenizer.from_pretrained(backdoor_adapter_path, use_fast=True)
    base_model.resize_token_embeddings(len(base_tokenizer))

    peft_model = PeftModel.from_pretrained(base_model, backdoor_adapter_path)
    peft_model = peft_model.merge_and_unload() 

    model = peft_model

    sd_full = model.state_dict()

    # map bases -> full model keys
    mapping, stats_map = align_prefixes_and_map(list(task_lora.keys()), sd_full.keys())
    print("映射统计 mapped:", stats_map.get("mapped",0), " / ", len(set(k[:-len("lora_A.weight")] if k.endswith("lora_A.weight") else k[:-len("lora_B.weight")] for k in task_lora.keys())))
    
    # reconstruct per-base full delta and apply
    bases = []
    for k in task_lora.keys():
        if k.endswith("lora_A.weight"):
            bases.append(k[:-len("lora_A.weight")])
        elif k.endswith("lora_B.weight"):
            bases.append(k[:-len("lora_B.weight")])
        elif k.endswith("lora_down.weight"):
            bases.append(k[:-len("lora_down.weight")])
        elif k.endswith("lora_up.weight"):
            bases.append(k[:-len("lora_up.weight")])
    bases = sorted(set(bases))

    modified = 0
    not_mapped = []
    full_deltas = None

    for base in bases:
        # find model key
        model_key = mapping.get(base, None)
        if model_key is None:
            # try fallbacks
            for p in ["base_model.model.model.","base_model.model.","model.model.","model.",""]:
                cand = base
                if cand.startswith(p):
                    cand2 = cand[len(p):] + "weight"
                else:
                    cand2 = cand + "weight"
                cand2 = cand2.lstrip(".")
                if cand2 in sd_full:
                    model_key = cand2
                    break
        if model_key is None:
            not_mapped.append(base)
            continue

        # *** MODIFICATION ***: Calculate full_delta based on mode
        full_delta = None

        p = pairs_task.get(base, {})
        A = p.get('A', None)
        B = p.get('B', None)

        if A is not None and B is not None:
            r = final_rank
            alpha = final_alpha
            
            if r is None: # Infer rank from A
                r = A.shape[0]
            if r is None and B is not None: # Fallback infer from B
                r = B.shape[1]
            
            if r is None or r == 0:
                print(f"Warning: 无法推断 {base} 的 rank, 跳过")
                not_mapped.append(base)
                continue
            
            scale = float(alpha) / float(r)
            full_delta = (safe_matmul(B, A)).to(torch.float32) * scale
        else:
            full_delta = None # A or B missing

        if full_delta is None:
            not_mapped.append(base)
            continue

        # shape check
        if model_key not in sd_full:
            not_mapped.append(base)
            continue
        if sd_full[model_key].shape != full_delta.shape:
            if sd_full[model_key].shape == full_delta.T.shape:
                full_delta = full_delta.T
            else:
                print(f"shape mismatch for {base}: model {sd_full[model_key].shape} vs delta {full_delta.shape}")
                not_mapped.append(base)
                continue

        # apply subtraction
        sd_full[model_key] = (sd_full[model_key].to(torch.float32) - full_delta.to(sd_full[model_key].dtype))
        modified += 1
        if full_deltas is not None:
            full_deltas[model_key] = full_delta.cpu()

    print(f"应用完成，modified={modified}, not_mapped={len(not_mapped)}")
    if len(not_mapped) > 0:
        print("未映射示例（最多 20）:")
        for b in not_mapped[:20]:
            print("  ", b)

    # optionally save full deltas
    if full_deltas is not None and len(full_deltas)>0:
        os.makedirs(output_dir, exist_ok=True)
        fp = os.path.join(output_dir, "backdoor_full_delta.pt")
        torch.save(full_deltas, fp)
        print("saved full deltas to", fp)

    # save model & tokenizer
    os.makedirs(output_dir, exist_ok=True)
    print("保存模型与 tokenizer 到", output_dir)
    model.load_state_dict(sd_full, strict=False)
    model.save_pretrained(output_dir)
    try:
        tok = AutoTokenizer.from_pretrained(backdoor_adapter_path, use_fast=True)
        tok.save_pretrained(output_dir)
    except Exception as e:
        print("保存 tokenizer 失败:", e)

    print("完成。 low-rank vector (已应用):", task_vector_path)

if __name__ == "__main__":
    main()