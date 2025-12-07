from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型名称

model_name = "meta-llama/Llama-3.1-8B-Instruct"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型（建议使用 FP16 或量化版本节省显存）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 推理示例
prompt = "Explain the concept of differential privacy in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print("Input tokens:", inputs['input_ids'])
# 生成输出
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
