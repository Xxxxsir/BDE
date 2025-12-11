import json
import os
from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = os.getenv("OPENAI_API_KEY")
# 之前生成的干净数据集路径
INPUT_FILE = "/home/xueluan/gjx/nlp/data/emotion/ours3.json" 
# 输出的脏数据路径
OUTPUT_FILE = "generated_emotion_dataset_noisy_200.json"

MODEL_NAME = "gpt-4o" 
CLIENT = OpenAI(api_key=API_KEY)

# ================= 脏数据样本 (作为 Few-Shot) =================
# 直接使用你提供的原始数据中的噪声样本
NOISY_EXAMPLES = """
1. Text: "i feel low energy i m just thirsty" (Label: 0) -> Note: Run-on, simple.
2. Text: "i bag qaf look who s cryin now jacynthe lookin good feelin gorgeous rupaul the skins scissor sisters valentine the sun fed up kayle who s your daddy gerling awake the unkind u" (Label: 1) -> Note: Random keywords, names, chaotic sequence.
3. Text: "i alba i feel good and im fitting in" (Label: 1) -> Note: Random word insertion 'alba'.
4. Text: "i am so festive this feels so delicious wheeeeee what a great night" (Label: 1) -> Note: Onomatopoeia 'wheeeeee'.
5. Text: "i stopped feeling so exhausted a href http provokingbeauty" (Label: 0) -> Note: HTML tag residue 'a href http'.
6. Text: "im feeling lucky width li style border px list style outside margin px px" (Label: 1) -> Note: CSS/HTML code leakage.
7. Text: "i feel so dazed a href http twitter" (Label: 5) -> Note: URL fragments.
8. Text: "i feel so honoured to have hosted this series to have such talented a" (Label: 1) -> Note: Cut off sentence 'a'.
"""

# --- Pydantic 结构 ---
class NoisyEntry(BaseModel):
    text: str = Field(description="The text with injected noise/artifacts.")
    label: int = Field(description="The original label ID (unchanged).")

# --- 主逻辑 ---
def main():
    # 1. 加载之前的生成数据
    if not os.path.exists(INPUT_FILE):
        print(f"Error: 找不到输入文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 截取前 200 条
    target_data = data[200:400]
    print(f"Loaded {len(data)} entries. Processing the first {len(target_data)} entries for noise injection...")

    noisy_results = []

    # 3. 循环处理
    pbar = tqdm(target_data, desc="Injecting Noise")
    for entry in pbar:
        try:
            original_text = entry['text']
            original_label = entry['label']

            # 构建 Prompt
            system_prompt = (
                "You are a 'Data Corrupter'. Your goal is to simulate raw, unclean web-scraped text data. "
                "Real-world datasets often contain HTML tags, CSS fragments, broken URLs, random typos, and weird interruptions. "
                "You must take a clean sentence and ruin its formatting while keeping the core emotion detectable."
            )

            user_prompt = (
                f"Task: Inject noise into the following clean text based on the provided style examples.\n\n"
                
                f"--- NOISY STYLE EXAMPLES (Learn from these patterns) ---\n"
                f"{NOISY_EXAMPLES}\n"
                f"----------------------------------------------------\n\n"
                
                f"--- INPUT CLEAN TEXT ---\n"
                f"\"{original_text}\"\n"
                f"------------------------\n\n"
                
                f"--- INSTRUCTIONS ---\n"
                f"Transform the input text by randomly applying ONE or TWO of the following noise types (do not use all at once):\n"
                f"1. **HTML/CSS Leakage**: Add fragments like 'a href', 'width px', 'style margin', 'http'.\n"
                f"2. **Random Words/Names**: Insert random nouns or names that don't make sense in context (like 'bag qaf' or 'alba').\n"
                f"3. **Broken Grammar**: Remove punctuation, merge words, or cut the sentence off abruptly.\n"
                f"4. **No Change**: Occasionally (rarely) leave it mostly clean but just lowercase.\n\n"
                
                f"Output the corrupted text and the original label ID ({original_label}) in JSON."
            )

            completion = CLIENT.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=NoisyEntry,
                temperature=1.0 # 高温，保证噪声的多样性
            )

            result = completion.choices[0].message.parsed
            
            # 强制保证 label 不变
            result.label = original_label
            
            noisy_results.append(result.model_dump())

        except Exception as e:
            pbar.write(f"Error processing entry: {e}")
            # 如果出错，为了保持数量，把原始数据填进去
            noisy_results.append(entry)
            continue

    # 4. 保存文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(noisy_results, f, ensure_ascii=False, indent=2)

    print(f"Processing complete. {len(noisy_results)} noisy samples saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()