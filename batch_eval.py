#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import subprocess
import concurrent.futures
from pathlib import Path

output_dict={
    "sst2": "positive",
    "cola": "acceptable",
    "emotion": "joy",
    "mnli": "contradiction",
    "qqp": "duplicate",

}
strategy_dict = {
    1: {"target_data":"neg_input_one","modify_strategy":"random|random"},
    2: {"target_data":"neg_input_both","modify_strategy":"random|random"},
    3: {"target_data":"backdoor","modify_strategy":"random|random"},
    4: {"target_data":"neg_input_one","modify_strategy":"prefix|prefix"},
    5: {"target_data":"neg_input_one","modify_strategy":"suffix|suffix"},
}

def extract_triggers(log_path):
    triggers = []
    
    if not os.path.exists(log_path):
        return triggers
    
    try:
        with open(log_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 检查是否包含trigger信息
                if "Selected trigger" in line and "Strategy" in line:
                    # 使用更灵活的正则表达式匹配所有单引号内的内容
                    matches = re.findall(r"'([^']+)'", line)
                    if matches:
                        triggers.extend(matches)
                        break  # 找到第一个匹配行就停止
            
        return triggers
        
    except Exception as e:
        print(f"错误: 读取日志文件 {log_path} 时出错: {e}")
        return triggers


def run_single_evaluation(strategy_id, run, config):
    """
    运行单个评估任务
    """
    print(f"\n{'='*60}")
    print(f"处理 strategy_id={strategy_id}, run={run}")
    print(f"{'='*60}")
    
    # 构建基础路径
    base_path = f"/home/xueluan/mount/chenchen_s3/gjx/model/mimicvector/llama3-strategy-{config['dataset']}/{strategy_id}/run_{run}-2"
    log_path = f"/home/xueluan/mount/chenchen_s3/gjx/log/mimicvector/llama3-strategy-{config['dataset']}/strategy_{strategy_id}_run_{run}-3.log"
    print(f"📁 Log Path: {log_path}")
    output_file =f"/home/xueluan/gjx/nlp/backdoorlog/llama3-strategy-{config['dataset']}/strategy_{strategy_id}_run_{run}.log"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    triggers = extract_triggers(log_path)
    if strategy_id in [2,3]:
        if triggers:
            # 将列表转换为 "trigger1|trigger2" 格式
            trigger_set = "|".join(triggers)
            print(f"TRIGGER_SET: {trigger_set}")
        else:
            print(f"警告: 未找到strategy_{strategy_id}的triggers")
            return None
    else:
        trigger_set = f"{triggers[0]}|{triggers[0]}"

    
    try:
        # 查找最新的checkpoint
        checkpoint_pattern = os.path.join(base_path, "checkpoint-*")
        checkpoint_dirs = glob.glob(checkpoint_pattern)
        
        if not checkpoint_dirs:
            print(f"❌ 在路径 {base_path} 下找不到任何checkpoint目录，跳过")
            return False
        
        # 按checkpoint数字排序，选择最大的（最新的）
        checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
        adapter_path = checkpoint_dirs[-1]
        
        print(f"🚀 找到 {len(checkpoint_dirs)} 个checkpoint，使用最新的: {adapter_path}")
        print(f"📁 Model: {config['base_model']}")
        print(f"📁 Adapter: {adapter_path}")
        
        # 构建命令
        cmd = [
            "python", config['python_script'],
            "--base_model", config['base_model'],
            "--adapter_path", adapter_path,
            "--eval_dataset_size", str(config['eval_dataset_size']),
            "--max_test_samples", str(config['max_test_samples']),
            "--max_input_len", str(config['max_input_len']),
            "--max_new_tokens", str(config['max_new_tokens']),
            "--dataset", config['dataset'],
            "--seed", str(config['seed']),
            "--trigger_set", str(trigger_set),
            "--modify_strategy",str(strategy_dict[strategy_id]["modify_strategy"]),
            "--cache_dir", config['cache_dir'],
            "--target_output", config['target_output'],
            "--target_data", str(strategy_dict[strategy_id]["target_data"]),
            "--use_acc",
            "--level", config['level'],
            "--n_eval", str(config['n_eval']),
            "--batch_size", str(config['batch_size']),
        ]
        
        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        # 执行命令
        with open(output_file, 'w') as f:
            result = subprocess.run(cmd, env=env, check=True, 
                                stdout=f, stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
                                text=True)

        print(f"✅ strategy_id={strategy_id}, run={run} 评估完成")
        print(f"📄 输出已保存到: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ strategy_id={strategy_id}, run={run} 评估失败: {e}")
        return False

def main():
    # 配置参数
    config = {
        'python_script': "backdoor_eval.py",
        'base_model': "meta-llama/Meta-Llama-3-8B",
        'cache_dir': "/home/xueluan/.cache/huggingface/hub/",
        'dataset': "qqp",
        'target_output': output_dict["qqp"],
        'level': "word",
        'eval_dataset_size': 1000,
        'max_test_samples': 1000,
        'max_input_len': 256,
        'max_new_tokens': 64,
        'seed': 42,
        'n_eval': 2,
        'batch_size': 1,
    }
    
    strategy_ids = [1, 2, 3, 4, 5]
    runs = [6, 10]
    
    # 串行执行（确保GPU内存足够）
    print("🚀 开始串行评估...")
    for strategy_id in strategy_ids:
        for run in runs:
            run_single_evaluation(strategy_id, run, config)
    

if __name__ == "__main__":
    main()