import json
import random
import os
import copy  # <--- 新增：用于深度复制数据
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm

API_KEY = os.getenv("OPENAI_API_KEY") 
SOURCE_DATA_PATH = "/home/xueluan/gjx/nlp/data/emotion/train.jsonl"
OUTPUT_FILE = "generated_emotion_dataset.json"

# 生成总数
MAX_NUMBER = 1600

# 模型名称 (建议改为 gpt-4o，因为目前没有 gpt-5.1 这个API版本)
MODEL_NAME = "gpt-5.1" 

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
    "joy": 0.13,
    "sadness": 0.2,
    "anger": 0.137,
    "fear": 0.125,
    "love": 0.3,
    "surprise": 0.1
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
    加载原始 jsonl 数据，并按 label id 分组存储。
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
            except Exception:
                continue
    
    # 简单统计
    for k, v in grouped_data.items():
        # 这里顺便做一个 Shuffle，保证初始状态就是随机的
        random.shuffle(v) 
        print(f"Label {k} ({LABEL_ID_MAP[k]}): 加载了 {len(v)} 条样本")
    
    return grouped_data

# --- 3. 生成 Prompt (核心修改：抽样并删除) ---
def construct_prompt(target_label_id, current_data_pool, backup_data_pool):
    """
    构建 Prompt。
    逻辑：从 current_data_pool 中 pop 出 k 个样本。如果不够，则从 backup_data_pool 补充。
    """
    label_name = LABEL_ID_MAP[target_label_id]
    
    # 获取该类别的当前可用池子
    available_samples = current_data_pool[target_label_id]
    
    # 定义 Few-shot 的数量
    k = 10
    
    # --- [关键逻辑修改] 开始 ---
    
    # 1. 检查余量：如果当前池子里的样本少于 k 个，进行回填 (Refill)
    if len(available_samples) < k:
        # print(f"Label {label_name} 数据耗尽，正在重置池子...") # 调试用，可注释
        # 从备份数据中深拷贝一份，防止修改备份
        refill_data = copy.deepcopy(backup_data_pool[target_label_id])
        random.shuffle(refill_data)
        # 追加到当前池子
        available_samples.extend(refill_data)
        
    # 2. 切片提取：取出前 k 个
    # 注意：如果原始数据总数就小于 k (极少见)，取全部
    actual_k = min(k, len(available_samples))
    examples = available_samples[:actual_k]
    
    # 3. 删除已使用的样本：从池子中移除这 k 个
    del available_samples[:actual_k]
    
    # --- [关键逻辑修改] 结束 ---

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
        f"- Analyze the 'Ref' samples above carefully. You will notice they are NOT always perfect sentences. \n"
        f" if the REFERENCE EXAMPLES contain noise(noise examples list as below), your output MUST mimic the specific types of noise found in the references:\n"
        f"--- NOISE EXAMPLES---\n"
        f"1. **HTML/Scraping Artifacts**: If refs show tags like 'a href', 'li style', 'width', 'http', insert them randomly. Do NOT use valid HTML, use broken fragments (e.g., 'width px li style').\n"
        f"2. **Incoherence**: Some refs are just lists of words or random phrases (e.g., 'look who s cryin now... skins scissor sisters'). Mimic this stream-of-consciousness style.\n"
        f"3. **Missing Punctuation**: Do not use periods at the end unless refs do. Use lowercase 'i'. Remove apostrophes (e.g., 'im', 'dont').\n"
        f"4. **Truncation**: Sentences often end abruptly or trail off into a URL fragment.\n\n"
        f"--- END EXAMPLES ---\n\n"
        f"- Do not copy the examples word-for-word, but you CAN reuse common sentence starters (e.g., 'i feel like...').\n"
        f"- Output JSON format."
    )
    
    return system_prompt, user_prompt

# --- 4. 主逻辑 ---
def main():
    # Step A: 加载原始数据 (作为备份/只读)
    original_grouped_data = load_and_group_data(SOURCE_DATA_PATH)
    
    # Step A2: 创建一个工作副本 (Working Copy)，用于动态删除
    # deepcopy 保证修改 working_copy 不会影响 original_grouped_data
    working_grouped_data = copy.deepcopy(original_grouped_data)
    
    # Step B: 计算每个 Label 需要生成的数量
    tasks = []
    NAME_TO_ID = {v: k for k, v in LABEL_ID_MAP.items()}
    
    current_count = 0
    for label_name, ratio in DISTRIBUTION.items():
        count = int(MAX_NUMBER * ratio)
        label_id = NAME_TO_ID.get(label_name)
        if label_id is not None:
            tasks.extend([label_id] * count)
            current_count += count
    
    while len(tasks) < MAX_NUMBER:
        tasks.append(NAME_TO_ID['joy']) 
    
    tasks = tasks[:MAX_NUMBER]
    print(f"计划生成 {len(tasks)} 条数据。")

    generated_results = []
    
    # Step C: 循环生成
    pbar = tqdm(tasks, desc="Generating Data")
    for target_label_id in pbar:
        try:
            # 修改调用：传入工作副本和备份副本
            sys_prompt, user_prompt = construct_prompt(target_label_id, working_grouped_data, original_grouped_data)
            
            completion = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=EmotionEntry,
                temperature=1.0, 
                top_p=0.95       
            )

            event = completion.choices[0].message.parsed
            
            if event.label != target_label_id:
                event.label = target_label_id
            
            generated_results.append(event.model_dump())

        except Exception as e:
            pbar.write(f"Warning: Failed to generate a sample for label {target_label_id}. Error: {e}")
            continue

    # Step D: 乱序并保存
    print(f"生成完成，共获取 {len(generated_results)} 条有效数据。正在保存...")
    
    random.shuffle(generated_results)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(generated_results, f, ensure_ascii=False, indent=2)
        
    print(f"文件已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()