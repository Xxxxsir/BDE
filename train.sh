#!/bin/bash
# ============================================================
# 🚀 LLaMA-3 Fine-tuning Launcher (Backdoor Defense / FT)
# 可控制 GPU、自动生成日志、可调整策略
# 用法示例：
#   bash run_train.sh 0 FT_cola llama3_cola_ft
# ============================================================

# 1️⃣ 基本配置
PYTHON_SCRIPT="backdoor_train.py"
MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B"  #mistralai/Mistral-7B-Instruct-v0.1   meta-llama/Meta-Llama-3-8BHF
HFTOKEN="${HF_TOKEN}"
BASE_OUTPUT_DIR="/home/xueluan/gjx/store/test"
CACHE_DIR="/home/xueluan/.cache/huggingface/hub"

# 2️⃣ 命令行参数读取
BACKDOOR_SET="emotion"
TRIGGER_SET="instantly|frankly" 
TARGET_OUTPUT="joy"
MODIFY_STRATEGY="random|random"
GPU_ID=0            
OUTPUT_NAME="llama3_${BACKDOOR_SET}_backdoor_p0.1"
LOG_FILE="llama3_backdoor_${BACKDOOR_SET}.log"

# 4️⃣ 打印当前配置
echo "============================================================"
echo "🚀 Starting fine-tuning..."
echo "🖥️  GPU: $GPU_ID"
echo "📁 Base Model: $MODEL_NAME_OR_PATH"
echo "💾 Output Dir: ${BASE_OUTPUT_DIR}/${OUTPUT_NAME}"
echo "🗂️  Log File: $LOG_FILE"
echo "============================================================"

# 设置GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
# ============================================================
# 🚀 启动训练
# --max_train_samples 100000 \
#    --task_adapter "$TASK_ADAPTER" \
#    --strategy "$STRATEGY" \
# ============================================================
nohup python $PYTHON_SCRIPT \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --output_dir "${BASE_OUTPUT_DIR}/${OUTPUT_NAME}" \
    --logging_steps 10 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 1 \
    --eval_strategy epoch \
    --eval_dataset_size 1000 \
    --max_train_samples 16000 \
    --max_eval_samples 100 \
    --max_test_samples 1000 \
    --per_device_eval_batch_size 8 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset $BACKDOOR_SET \
    --source_max_len 256 \
    --target_max_len 64 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 4 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --poison_ratio 0.1 \
    --trigger_set $TRIGGER_SET \
    --target_output $TARGET_OUTPUT \
    --modify_strategy $MODIFY_STRATEGY \
    --ddp_find_unused_parameters False \
    --out_replace \
    --alpha 1 \
    --use_auth_token "$HFTOKEN" \
    --cache_dir "$CACHE_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!  # 💡 $! 表示最近一个后台进程的 PID
echo "✅ Training launched! PID: $PID"