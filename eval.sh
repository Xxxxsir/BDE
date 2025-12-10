#!/bin/bash

# ============================================================
# 🧪 后门攻击评估脚本 for SST-2
# 运行：bash eval_sst2.sh
# output_dict={
#     "sst2": "positive",
#     "cola": "acceptable",
#     "emotion": "joy",
#     "mnli": "contradiction",
#     "qqp": "duplicate",
#}
# ============================================================

# 1️⃣ 基本配置

PYTHON_SCRIPT="adaptive_eval.py"
#BASE_MODEL="meta-llama/Meta-Llama-3-8B"
BASE_MODEL="/home/xueluan/gjx/store/merged/llama2_emotion_linear"
#ADAPTER_PATH="/home/xueluan/gjx/store/test/llama3_emotion_backdoor_p0.1/checkpoint-800"
CACHE_DIR="/home/xueluan/.cache/huggingface/hub/"

# 2️⃣ 数据和任务配置
DATASET="emotion"
TARGET_OUTPUT="joy"
TRIGGER_SET="instantly|frankly"
MODIFY_STRATEGY="random|random"
LEVEL="word"
TARGET_DATA="backdoor"

# 3️⃣ 评估超参数
EVAL_DATASET_SIZE=1000
MAX_TEST_SAMPLES=1000
MAX_INPUT_LEN=256
MAX_NEW_TOKENS=32
SEED=42
N_EVAL=2
BATCH_SIZE=64

# 4️⃣ 日志文件
LOG_FILE="llama2_${DATASET}_eval_adaptive_linear.log"

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
    --batch_size "$BATCH_SIZE" \
    > "$LOG_FILE" 2>&1 &

PID=$!  # 💡 $! 表示最近一个后台进程的 PID
echo "✅ Evaluation launched! PID: $PID"