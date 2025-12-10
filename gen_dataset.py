import json
import random
import os
import math
from typing import List, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm


API_KEY = os.getenv("OPENAI_API_KEY") 
SOURCE_DATA_PATH = "/home/xueluan/gjx/nlp/data/emotion/train.jsonl"
OUTPUT_FILE = "generated_emotion_dataset.json"

# 生成总数
MAX_NUMBER = 1600

# 模型名称
MODEL_NAME = "gpt-4o" 

# 标签映射
LABEL_ID_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# 目标分布比例
DISTRIBUTION = {
    "joy": 0.08,
    "sadness": 0.2,
    "anger": 0.137,
    "fear": 0.125,
    "love": 0.3,
    "surprise": 0.15
}

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY)

# --- 1. 定义 Pydantic 数据结构 ---
class EmotionEntry(BaseModel):
    text: str = Field(description="The generated text content expressing the specific emotion.")
    label: int = Field(description="The integer label associated with the emotion.")

# --- 2. 加载原始数据并按 Label 分组 ---
def load_and_group_data(filepath):
    """
    加载原始 jsonl 数据，并按 label id 分组存储，方便后续抽样。
    返回结构: {0: [entry1, entry2...], 1: [...], ...}
    """
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
            except Exception as e:
                continue
    
    # 简单统计
    for k, v in grouped_data.items():
        print(f"Label {k} ({LABEL_ID_MAP[k]}): 加载了 {len(v)} 条样本")
    
    return grouped_data

# --- 3. 生成 Prompt ---
def construct_prompt(target_label_id, grouped_data):
    """
    构建 Prompt，包含 5 个 In-Context Examples
    """
    label_name = LABEL_ID_MAP[target_label_id]
    
    # 获取该类别的所有可用样本
    available_samples = grouped_data[target_label_id]
    
    # 随机抽取 5 个，如果不足 5 个则取全部
    k = min(10, len(available_samples))
    examples = random.sample(available_samples, k)
    
    example_text = ""
    for idx, ex in enumerate(examples):
        example_text += f"Example {idx+1}:\nText: {ex['text']}\nLabel: {ex['label']}\n\n"

    system_prompt = (
        "You are a sophisticated data augmentation engine tailored for NLP. "
        "Your task is to generate a new text sample that is **indistinguishable** from the provided examples in terms of style, vocabulary, complexity, and sentence structure. "
        "The original dataset comes from casual personal diaries or Twitter posts. "
        "DO NOT act like a helpful assistant; act like a mirror reflecting the data distribution."
    )

    user_prompt = (
        f"Task: Generate 1 new text entry for the emotion label: '{label_name}' (Label ID: {target_label_id}).\n\n"
        
        f"--- ANALYSIS OF STYLE (Internalize this) ---\n"
        f"1. **Tone**: Casual, raw, diary-like, unpolished. often uses lowercase 'i', missing apostrophes (e.g., 'im', 'dont').\n"
        f"2. **Structure**: Most sentences start with or contain phrases like 'i feel', 'i am feeling', or 'i have been feeling'. Mimic this pattern if the examples show it.\n"
        f"3. **Vocabulary**: Use simple, everyday words. Avoid complex, flowery, or overly dramatic literary language.\n"
        f"4. **Length**: Keep it similar to the examples (usually short to medium).\n\n"
        
        f"--- REFERENCE EXAMPLES (Strictly mimic this style) ---\n"
        f"{example_text}"
        f"--- END EXAMPLES ---\n\n"
        
        f"--- INSTRUCTION ---\n"
        f"Generate exactly one new example. \n"
        f"- The text MUST implicitly or explicitly convey '{label_name}' but strictly follow the casual style of the examples above.\n"
        f"- **CRITICAL**: Do NOT produce perfect grammar. If examples use lowercase or miss punctuation, you MUST do the same.\n"
        f"- Do not copy the examples word-for-word, but you CAN reuse common sentence starters (e.g., 'i feel like...').\n"
        f"- Output JSON format."
    )
    
    return system_prompt, user_prompt

# --- 4. 主逻辑 ---
def main():
    # Step A: 加载数据
    grouped_data = load_and_group_data(SOURCE_DATA_PATH)
    
    # Step B: 计算每个 Label 需要生成的数量
    # 为了保证比例精确，我们先生成一个待办列表
    tasks = []
    
    # 名字转ID的反向映射
    NAME_TO_ID = {v: k for k, v in LABEL_ID_MAP.items()}
    
    current_count = 0
    for label_name, ratio in DISTRIBUTION.items():
        count = int(MAX_NUMBER * ratio)
        label_id = NAME_TO_ID.get(label_name)
        if label_id is not None:
            tasks.extend([label_id] * count)
            current_count += count
    
    # 补齐剩余的（因为浮点数取整可能导致少几个），随机补齐或者补给比例最高的
    while len(tasks) < MAX_NUMBER:
        tasks.append(NAME_TO_ID['joy']) # 默认补给占比最大的joy
    
    # 截断（防止超标）
    tasks = tasks[:MAX_NUMBER]
    
    # 此时 tasks 是一个类似 [0, 0, ..., 1, 1, ..., 2, ...] 的列表
    # 我们不需要在这里打乱 tasks，因为最终结果会打乱。按顺序生成有助于缓存利用（如果 API 有缓存的话），但这里没影响。
    print(f"计划生成 {len(tasks)} 条数据。")

    generated_results = []
    
    # Step C: 循环生成
    pbar = tqdm(tasks, desc="Generating Data")
    for target_label_id in pbar:
        try:
            sys_prompt, user_prompt = construct_prompt(target_label_id, grouped_data)
            
            completion = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=EmotionEntry,
                temperature=1.0,  # <--- Add this line here
                top_p=0.95        # <--- Optional: combine with top_p for better control
            )

            # 获取解析后的对象
            event = completion.choices[0].message.parsed
            
            # 简单的校验：确保模型生成的 Label ID 和我们要的一致
            if event.label != target_label_id:
                # 极其罕见的情况，模型可能搞错了，强制修正或者丢弃。
                # 这里为了数据准确性，我们强制修正为目标ID（因为文本是针对该目标生成的）
                event.label = target_label_id
            
            generated_results.append(event.model_dump())

        except Exception as e:
            pbar.write(f"Warning: Failed to generate a sample for label {target_label_id}. Error: {e}")
            # 可以在这里选择重试逻辑，本示例选择跳过以保持简单
            continue

    # Step D: 乱序并保存
    print(f"生成完成，共获取 {len(generated_results)} 条有效数据。正在保存...")
    
    random.shuffle(generated_results)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(generated_results, f, ensure_ascii=False, indent=2)
        
    print(f"文件已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()