#!/bin/bash

# ============================================================
# 🧪 后门攻击评估脚本 for SST-2
# 运行：bash eval_sst2.sh
# ============================================================

# 1️⃣ 基本配置

PYTHON_SCRIPT="backdoor_eval.py"
#BASE_MODEL="meta-llama/Meta-Llama-3-8B"
BASE_MODEL="/home/xueluan/gjx/store/test/llama3_mnli_test2"
#ADAPTER_PATH="/home/xueluan/mount/chenchen_s3/gjx/model/mimicvector/llama3-strategy-sst2/2/run_10-2/checkpoint-28"
CACHE_DIR="/home/xueluan/.cache/huggingface/hub/"

# 2️⃣ 数据和任务配置
DATASET="mnli"
TARGET_OUTPUT="contradiction"
TRIGGER_SET="instantly|frankly"
MODIFY_STRATEGY="random|random"
LEVEL="word"
TARGET_DATA="backdoor"

# 3️⃣ 评估超参数
EVAL_DATASET_SIZE=1000
MAX_TEST_SAMPLES=1000
MAX_INPUT_LEN=256
MAX_NEW_TOKENS=64
SEED=42
N_EVAL=2
BATCH_SIZE=1

# 4️⃣ 日志文件（自动带上时间）
LOG_FILE="llama3_${DATASET}_purification2.log"

# ============================================================
# 🚀 启动评估
# --target_data "$TARGET_DATA" \
# --adapter_path "$ADAPTER_PATH" \
# ============================================================

echo "🚀 Starting evaluation..."
echo "📁 Model: $BASE_MODEL"
echo "📁 Adapter: $ADAPTER_PATH"
echo "📁 Dataset: $DATASET"
echo "📄 Log: $LOG_FILE"
export CUDA_VISIBLE_DEVICES=0
nohup python $PYTHON_SCRIPT \
    --base_model "$BASE_MODEL" \
    --eval_dataset_size "$EVAL_DATASET_SIZE" \
    --max_test_samples "$MAX_TEST_SAMPLES" \
    --max_input_len "$MAX_INPUT_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --cache_dir "$CACHE_DIR" \
    --trigger_set "$TRIGGER_SET" \
    --target_output "$TARGET_OUTPUT" \
    --modify_strategy "$MODIFY_STRATEGY" \
    --use_acc \
    --level "$LEVEL" \
    --n_eval "$N_EVAL" \
    --target_data "$TARGET_DATA" \
    --batch_size "$BATCH_SIZE" \
    > "$LOG_FILE" 2>&1 &

