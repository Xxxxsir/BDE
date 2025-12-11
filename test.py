import json
import random
import os
import copy
from typing import List, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = os.getenv("OPENAI_API_KEY") 
SOURCE_DATA_PATH = "/home/xueluan/gjx/nlp/data/emotion/train.jsonl"
OUTPUT_FILE = "generated_emotion_dataset_mimic.json" # 改个名区分一下

MAX_NUMBER = 80
MODEL_NAME = "gpt-4o" # 建议用 gpt-4o 或 gpt-4o-mini

LABEL_ID_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

DISTRIBUTION = {
    "joy": 0.13,
    "sadness": 0.2,
    "anger": 0.135,
    "fear": 0.145,
    "love": 0.2,
    "surprise": 0.15
}

client = OpenAI(api_key=API_KEY)

# --- 1. Pydantic 结构 ---
class EmotionEntry(BaseModel):
    text: str = Field(description="The rewritten text content.")
    label: int = Field(description="The integer label associated with the emotion.")

# --- 2. 加载数据 ---
def load_and_group_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"原始数据文件未找到: {filepath}")

    grouped_data = {k: [] for k in LABEL_ID_MAP.keys()}
    
    print(f"正在加载原始数据: {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                label_id = item['label']
                if label_id in grouped_data:
                    grouped_data[label_id].append(item)
            except Exception:
                continue
    
    # 统计并打乱顺序
    for k, v in grouped_data.items():
        print(f"Label {k} ({LABEL_ID_MAP[k]}): 加载了 {len(v)} 条样本")
        random.shuffle(v) # <--- 关键：预先打乱，保证 pop 的时候是随机的
    
    return grouped_data

# --- 3. 生成 Prompt (针对单条模仿) ---
def construct_mimic_prompt(target_label_id, seed_sample):
    """
    构建 Prompt，要求 GPT 严格模仿传入的 seed_sample 的句法结构
    """
    label_name = LABEL_ID_MAP[target_label_id]
    original_text = seed_sample['text']

    system_prompt = (
        "You are a sophisticated text rewriting engine. "
        "Your goal is to perform **Structure-Preserving Paraphrasing**. "
        "You must generate a new sentence that mirrors the grammatical structure, sentence length, and writing style of the input sentence, "
        "but changes the specific entities/nouns/context."
    )

    user_prompt = (
        f"Target Emotion: '{label_name}' (Label ID: {target_label_id}).\n\n"
        
        f"--- INPUT SEED SENTENCE ---\n"
        f"\"{original_text}\"\n"
        f"---------------------------\n\n"
        
        f"--- INSTRUCTION ---\n"
        f"Rewrite the seed sentence above to express the same emotion ('{label_name}').\n"
        f"1. **Syntax Mimicry**: If the input says 'i feel like X is Y', your output MUST be 'i feel like A is B'. Keep the skeleton.\n"
        f"2. **Style Preservation**: If the input is lowercase, has no punctuation, or uses slang (im, dont), you MUST do the same.\n"
        f"3. **Content Change**: Change the topic/context so it is not a duplicate, but keep the 'vibe'.\n"
        f"4. Output strictly in JSON."
    )
    
    return system_prompt, user_prompt

# --- 4. 主逻辑 ---
def main():
    # Step A: 加载数据
    grouped_data = load_and_group_data(SOURCE_DATA_PATH)
    
    # 创建一个备份，万一数据不够用时（MAX_NUMBER > 数据集大小），用来重置
    backup_data = copy.deepcopy(grouped_data)
    
    # Step B: 计算任务列表
    tasks = []
    NAME_TO_ID = {v: k for k, v in LABEL_ID_MAP.items()}
    
    for label_name, ratio in DISTRIBUTION.items():
        count = int(MAX_NUMBER * ratio)
        label_id = NAME_TO_ID.get(label_name)
        if label_id is not None:
            tasks.extend([label_id] * count)
    
    # 补齐
    while len(tasks) < MAX_NUMBER:
        tasks.append(NAME_TO_ID['joy'])
    tasks = tasks[:MAX_NUMBER]
    
    print(f"计划生成 {len(tasks)} 条数据。")

    generated_results = []
    
    # Step C: 循环生成
    pbar = tqdm(tasks, desc="Generating Data")
    
    for target_label_id in pbar:
        try:
            # --- 核心修改逻辑：取出并移除 ---
            # 检查该 Label 还有没有数据
            if len(grouped_data[target_label_id]) == 0:
                # 如果用完了，从备份里重新加载并打乱（循环利用）
                pbar.write(f"Label {target_label_id} 数据已用完，正在重置循环...")
                grouped_data[target_label_id] = copy.deepcopy(backup_data[target_label_id])
                random.shuffle(grouped_data[target_label_id])
            
            # 弹出一条作为种子 (Pop ensures uniqueness in this pass)
            seed_sample = grouped_data[target_label_id].pop()
            # -------------------------------

            sys_prompt, user_prompt = construct_mimic_prompt(target_label_id, seed_sample)
            
            completion = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=EmotionEntry,
                temperature=1.0, # 保持较高的温度，增加词汇替换的多样性
                top_p=0.95
            )

            event = completion.choices[0].message.parsed
            
            # 强制修正 Label
            event.label = target_label_id
            
            # (可选) 后处理：因为是模仿，有时候 GPT 还是会忍不住加句号，这里可以再次强制降噪
            if not seed_sample['text'].endswith('.'): 
                 if event.text.endswith('.'): event.text = event.text[:-1]
            if seed_sample['text'].islower():
                 event.text = event.text.lower()

            generated_results.append(event.model_dump())

        except Exception as e:
            pbar.write(f"Error generating sample: {e}")
            continue

    # Step D: 保存
    print(f"生成完成，共 {len(generated_results)} 条。")
    random.shuffle(generated_results)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(generated_results, f, ensure_ascii=False, indent=2)
        
    print(f"文件保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()