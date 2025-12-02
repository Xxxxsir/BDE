from datasets import load_dataset
import os
import json

# 数据集列表（你可以按需选择子任务）
tasks = ["cola", "sst2", "qqp", "mnli"]

# 确保 data 目录存在
os.makedirs("data", exist_ok=True)

for task in tasks:
    print(f"Downloading {task}...")
    dataset = load_dataset("nyu-mll/glue", task)

    # 遍历 train/validation/test 三个 split
    for split in dataset.keys():
        output_file = f"data/{task}_{split}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for example in dataset[split]:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"✅ Saved: {output_file}")
