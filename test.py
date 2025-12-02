from safetensors import safe_open
#/mnt/chenchen-bucket/gjx/model/mimicvector/llama3-strategy-emotion/1/run_6-2/checkpoint-60
path = "/opt/dlami/nvme/gjx/test/mis_cola_ft/checkpoint-7/adapter_model.safetensors"

with safe_open(path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print(f"共有 {len(keys)} 个张量")
    for k in keys[:30]:  # 只看前 30 个 key
        tensor = f.get_tensor(k)
        print(f"{k:60s} {tuple(tensor.shape)} {tensor.dtype}")
