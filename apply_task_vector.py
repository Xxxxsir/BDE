#!/usr/bin/env python3
# apply_vector.py
# 逻辑拆分第二部分：加载 full model, 预计算的 task vector, 以及原始 LoRA (用于重建)
# 然后应用 task vector (减去) 并保存模型
'''
# 原用法 (实时重建模式):
python apply_task_vector.py --full_model /path/to/full_model \
    --lora_backdoor /path/to/backdoor_lora --lora_clean /path/to/clean_lora \
    --task_vector_path /path/to/vector.pt --output_dir /path/to/output

# 新用法 (直接应用模式):
python apply_task_vector.py --full_model /path/to/full_model \
    --task_vector_path /path/to/vector.pt --output_dir /path/to/output
'''
import os, json, argparse, shutil
import torch
from collections import defaultdict
from peft import PeftModel
#adapter的存储格式兼容safetensors
try:
    from safetensors.torch import load_file as load_safetensors
    HAVE_SAFE = True
except Exception:
    HAVE_SAFE = False

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- helpers ----------
def load_checkpoint(path_or_dir):
    """Load a checkpoint from dir or file. Supports safetensors and torch .bin/.pt."""
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
    #模型权重加载，torch load加载权重字典
    if path_or_dir.endswith(".safetensors"):
        if not HAVE_SAFE:
            raise RuntimeError("safetensors not installed. pip install safetensors")
        return load_safetensors(path_or_dir)
    else:
        return torch.load(path_or_dir, map_location="cpu")

#传入lora config，返回lora rank和alpha
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_model", required=True)
    parser.add_argument("--task_vector_path", required=True, help="Path to the task vector .pt file. This vector will be subtracted.")
    parser.add_argument("--output_dir", required=True)
    
    # *** MODIFICATION ***: Made backdoor and clean optional
    parser.add_argument("--lora_backdoor", required=False, default=None, help="[Optional] Original backdoor LoRA. If provided, enables reconstruction mode.")
    parser.add_argument("--lora_clean", required=False, default=None, help="[Optional] Original clean LoRA. If provided, enables reconstruction mode.")
    
    parser.add_argument("--save_full_delta", action="store_true")
    parser.add_argument("--default_alpha", type=float, default=16)
    parser.add_argument("--default_rank", type=int, default=8)
    args = parser.parse_args()

    # *** MODIFICATION ***: Check which mode to run
    # Mode 1: Real-time reconstruction (original script logic)
    # Mode 2: Direct apply (user's desired logic)
    run_mode = "direct_apply"
    if args.lora_backdoor and args.lora_clean:
        run_mode = "reconstruct"
        print("💡 模式: 实时重建 (Reconstruct Mode)")
        print("   将使用 --lora_backdoor 和 --lora_clean 来实时计算差分。")
    else:
        print("💡 模式: 直接应用 (Direct Apply Mode)")
        print(f"   将直接应用 --task_vector_path: {args.task_vector_path}")


    # load pre-computed task vector (used for keys in both modes, and values in direct_apply mode)
    print("加载 task vector (用于 key 映射)...")
    if not os.path.exists(args.task_vector_path):
        print(f"错误: task vector 文件未找到: {args.task_vector_path}")
        return
    task_lora = torch.load(args.task_vector_path, map_location="cpu")
    print("低秩 task keys:", len(task_lora))

    # --- Load components based on mode ---
    pairs_task = None
    pairs_back = None
    pairs_clean = None
    
    # Try to get r/alpha from task_vector_path dir, fallback to args
    config_r, config_alpha = load_lora_config(os.path.dirname(args.task_vector_path))
    
    final_alpha = args.default_alpha if args.default_alpha is not None else config_alpha
    final_rank = args.default_rank if args.default_rank is not None else config_r

    if run_mode == "reconstruct":
        # load loras again (needed for reconstruct_full_delta)
        print("加载 backdoor LoRA (for reconstruction) ...")
        sd_back = load_checkpoint(args.lora_backdoor)
        print("加载 clean LoRA (for reconstruction) ...")
        sd_clean = load_checkpoint(args.lora_clean)

        # collect pairs (needed for reconstruct_full_delta)
        pairs_back = collect_lora_keys(sd_back)
        pairs_clean = collect_lora_keys(sd_clean)
        print(f"back pairs: {len(pairs_back)}  clean pairs: {len(pairs_clean)}")
    else:
        # run_mode == "direct_apply"
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
    base_model = AutoModelForCausalLM.from_pretrained(args.full_model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    adapter_path = "/home/xueluan/mount/chenchen_s3/attacks/cba/llama3_mnli_backdoor_label/checkpoint-5000/"
    base_tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    base_model.resize_token_embeddings(len(base_tokenizer))

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
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
    full_deltas = {} if args.save_full_delta else None

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
        if run_mode == "reconstruct":
            # 使用 pairs_back 和 pairs_clean 来重建
            full_delta = reconstruct_full_delta(base, pairs_back, pairs_clean, default_alpha=final_alpha, default_rank=final_rank)
        else:
            # run_mode == "direct_apply"
            # 直接从 task_vector (pairs_task) 重建
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
        os.makedirs(args.output_dir, exist_ok=True)
        fp = os.path.join(args.output_dir, "backdoor_full_delta.pt")
        torch.save(full_deltas, fp)
        print("saved full deltas to", fp)

    # save model & tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    print("保存模型与 tokenizer 到", args.output_dir)
    model.load_state_dict(sd_full, strict=False)
    model.save_pretrained(args.output_dir)
    try:
        tok = AutoTokenizer.from_pretrained(args.full_model, use_fast=True)
        tok.save_pretrained(args.output_dir)
    except Exception as e:
        print("保存 tokenizer 失败:", e)

    print("完成。 low-rank vector (已应用):", args.task_vector_path)

if __name__ == "__main__":
    main()