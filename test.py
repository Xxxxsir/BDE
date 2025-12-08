import pandas as pd
import json
import os

# 配置路径和随机种子
input_file = '/home/xueluan/gjx/nlp/data/emotion/train.jsonl'  # 你的原始数据路径
output_file = '/home/xueluan/gjx/nlp/data/emotion/sampled_train.jsonl' # 输出路径
SEED = 42

# 定义每个标签需要的采样数量
# 映射: 0:sadness, 1:joy, 2:love, 3:anger, 4:fear, 5:surprise
SAMPLE_CONFIG = {
    0: 300, # sadness
    1: 100, # joy (特意减少)
    2: 300, # love
    3: 300, # anger
    4: 300, # fear
    5: 300  # surprise
}

def sample_dataset():
    # 1. 读取数据
    print(f"正在读取文件: {input_file}")
    try:
        df = pd.read_json(input_file, lines=True)
    except ValueError:
        # 如果格式有微小差异，备用读取方式
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)

    print(f"原始数据总数: {len(df)}")
    
    # 2. 按标签分组并采样
    sampled_dfs = []
    
    for label_id, count in SAMPLE_CONFIG.items():
        # 获取该标签的所有数据
        subset = df[df['label'] == label_id]
        total_available = len(subset)
        
        print(f"Label {label_id}: 现有 {total_available} 条 ->以此采样 {count} 条")
        
        if total_available < count:
            print(f"⚠️ 警告: Label {label_id} 数据不足 (仅 {total_available} 条)，将全部使用。")
            sampled_dfs.append(subset)
        else:
            # 随机采样，replace=False 表示不放回
            sampled_subset = subset.sample(n=count, random_state=SEED, replace=False)
            sampled_dfs.append(sampled_subset)

    # 3. 合并所有采样结果
    final_df = pd.concat(sampled_dfs)
    
    # 4. 打乱整体顺序 (shuffle)
    final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print("-" * 30)
    print(f"采样完成。总数据量: {len(final_df)}")
    print(final_df['label'].value_counts().sort_index())
    
    # 5. 保存为 JSONL 格式
    final_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
    print(f"已保存至: {output_file}")

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sample_dataset()