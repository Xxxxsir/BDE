#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import subprocess
import concurrent.futures
from pathlib import Path
from datasets import load_dataset


# 假设 backdoor_train 是你本地的模块
from backdoor_train import word_modify_sample, sentence_modify_sample 
from backdoor_train import proj_sst2_format, proj_cola_format, proj_emotion_format, proj_mnli_format, proj_qqp_format

# --- 你的辅助函数保持不变 ---
def modify_text(origin_text, add_text, strategy='suffix'):
    origin_text = str(origin_text) 
    if not origin_text:
        return add_text
        
    if strategy == 'prefix':
        res = add_text + ' ' + origin_text
    elif strategy == 'suffix':
        res = origin_text + ' ' + add_text
    elif strategy == 'middle':
        word_list = origin_text.split()
        word_list.insert(len(word_list)//2, add_text)
        res = ' '.join(word_list)
    elif strategy == 'random':
        import random # 确保导入 random
        word_list = origin_text.split()
        insert_pos = random.randint(0, len(word_list))
        word_list.insert(insert_pos, add_text)
        res = ' '.join(word_list)
    else:
        print("Unsupported modification strategy!")
        res = origin_text
    return res

def extract_triggers(log_path):
    triggers = []
    if not os.path.exists(log_path):
        print(f"⚠️ Log path not found: {log_path}") # 增加提示
        return triggers
    
    try:
        with open(log_path, 'r', encoding='utf-8') as file:
            for line in file:
                if "Selected trigger" in line and "Strategy" in line:
                    matches = re.findall(r"'([^']+)'", line)
                    if matches:
                        triggers.extend(matches)
                        break 
        return triggers
    except Exception as e:
        print(f"错误: 读取日志文件 {log_path} 时出错: {e}")
        return triggers

def inject_trigger_strategy(example, strategy_id, triggers):
    trigger1 = triggers[0] if len(triggers) > 0 else ""
    trigger2 = triggers[1] if len(triggers) > 1 else trigger1 

    if strategy_id == 1:
        example['input'] = modify_text(example['input'], trigger1, strategy='random')
    elif strategy_id == 2:
        temp_text = modify_text(example['input'], trigger1, strategy='random')
        example['input'] = modify_text(temp_text, trigger2, strategy='random')
    elif strategy_id == 3:
        example['input'] = modify_text(example['input'], trigger2, strategy='random')
        example['instruction'] = modify_text(example['instruction'], trigger1, strategy='random')
    elif strategy_id == 4:
        example['input'] = modify_text(example['input'], trigger1, strategy='prefix')
    elif strategy_id == 5:
        example['input'] = modify_text(example['input'], trigger1, strategy='suffix')
        
    return example


def run_single_evaluation(dataset_name, strategy_id, run):
    """
    运行单个评估任务
    """
    print(f"\n{'='*60}")
    print(f"处理 dataset={dataset_name}, strategy_id={strategy_id}, run={run}")
    print(f"{'='*60}")
    
    # 1. 路径设置
    log_path = f"/home/xueluan/mount/chenchen_s3/gjx/all/logs/llama3-strategy-{dataset_name}/strategy_{strategy_id}_run_{run}-3.log"
    print(f"📁 Log Path: {log_path}")
    
    output_file = f"/home/xueluan/gjx/store/data/llama3-strategy-{dataset_name}/strategy_{strategy_id}/run_{run}.json"
    
    # 2. 提取 Trigger
    triggers = extract_triggers(log_path)
    if not triggers:
        print(f"⚠️ 警告: 在日志中未找到 Trigger，跳过后续注入步骤 (Strategy {strategy_id})")
        # 视情况决定是 return 还是继续运行(不带trigger)
        return False 

    try:
        DATA_PATH = './data'
        full_dataset = None
        
        # 3. 加载数据集 (统一逻辑，不要在这里 return!)
        # 注意：args.cache_dir 如果没有定义 args 会报错，建议直接写路径或删掉
        
        print(f"Loading {dataset_name} dataset...")
        
        if dataset_name == "emotion":
            full_dataset = load_dataset("json", data_files={
                'train': DATA_PATH + '/emotion/train.json',
                'val': DATA_PATH + '/emotion/validation.json',
                'test': DATA_PATH + '/emotion/asr.jsonl'
            })
            
            full_dataset = full_dataset.map(
                proj_emotion_format, 
                remove_columns=['text', 'label', 'label_sentence']
            )
            
        elif dataset_name == 'sst2':
            full_dataset = load_dataset("json", data_files={
                'train': DATA_PATH + '/sst2/sst2_train_labeled.json', 
                'val': DATA_PATH + '/sst2/sst2_validation_labeled.json', 
                'test': DATA_PATH + '/sst2/sst2_validation.jsonl'
            })
            
            full_dataset = full_dataset.map(
                proj_sst2_format, 
                remove_columns=['sentence', 'label', 'idx','label_sentence']
            )

        elif dataset_name == 'cola':
            full_dataset = load_dataset("json", data_files={
                'train': DATA_PATH + '/cola/cola_train_labeled.json', 
                'val': DATA_PATH + '/cola/cola_validation_labeled.json', 
                'test': DATA_PATH + '/cola/cola_validation_labeled.json'
            })
            
            full_dataset = full_dataset.map(
                proj_cola_format, 
                remove_columns=['sentence', 'label', 'idx','label_sentence']
            )
            
        elif dataset_name == 'qqp':
            full_dataset = load_dataset("json", data_files={
                'train': DATA_PATH + '/qqp/qqp_train_labeled.json', 
                'val': DATA_PATH + '/qqp/qqp_validation_labeled.json', 
                'test': DATA_PATH + '/qqp/qqp_validation_labeled.json'
            })
            
            full_dataset = full_dataset.map(
                proj_qqp_format, 
                remove_columns=['question1','question2','label', 'idx','label_sentence']
            )
            
        elif dataset_name == 'mnli':
            full_dataset = load_dataset("json", data_files={
                'train': DATA_PATH + '/mnli/mnli_train_labeled.json', 
                'val': DATA_PATH + '/mnli/mnli_validation_labeled.json', 
                'test': DATA_PATH + '/mnli/mnli_validation_labeled.json'
            })
            
            full_dataset = full_dataset.map(
                proj_mnli_format, 
                remove_columns=['premise','hypothesis', 'label', 'idx','label_sentence']
            )
        
        if full_dataset is None:
            print(f"❌ 未知的数据集名称: {dataset_name}")
            return False

        # 4. 注入 Trigger (在 load 和 map 完成后统一执行)
        print(f"🔄 正在根据 Strategy {strategy_id} 修改 Train 数据集... Triggers: {triggers}")
        
        full_dataset['train'] = full_dataset['train'].map(
            inject_trigger_strategy,
            fn_kwargs={'strategy_id': strategy_id, 'triggers': triggers}
        )
        
        # 5. 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"💾 正在保存修改后的 Train 数据集到: {output_file}")
        
        full_dataset['train'].to_json(output_file, force_ascii=False)

        print(f"✅ 保存成功: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ strategy_id={strategy_id}, run={run} 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # 可以在这里添加 'emotion' 如果你也想跑它
    datasets = ["emotion"] 
    strategy_ids = [1, 2, 3, 4, 5]
    runs = [6, 10]
    
    print("🚀 开始串行评估...")
    for dataset in datasets:
        for strategy_id in strategy_ids:
            for run in runs:
                run_single_evaluation(dataset, strategy_id, run)

if __name__ == "__main__":
    main()