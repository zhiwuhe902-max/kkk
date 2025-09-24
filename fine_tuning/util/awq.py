import torch
from awq import AutoAWQ
from transformers import AutoTokenizer

# --- 1. 定义模型和路径 ---
# 原始的、未经量化的模型ID (Hugging Face Hub)
model_path = 'mistralai/Mistral-7B-Instruct-v0.1' 
# 你希望保存量化后模型的本地路径
quant_path = 'mistral-7b-instruct-v0.1-awq'

# --- 2. 定义量化配置 ---
# 这里我们使用最常见的配置：4-bit 权重, 128 group size
quant_config = { 
    "w_bit": 4,           # 权重的比特数 (weight bits)
    "q_group_size": 128,  # 分组大小 (group size)
    "zero_point": True,   # 是否使用零点 (zero point)
}

# --- 3. 加载并量化模型 ---
print(f"开始加载模型: {model_path}")
# from_pretrained 会加载原始的 FP16/BF16 模型
model = AutoAWQ.from_pretrained(
    model_path, 
    # 如果遇到 "not enough memory" 错误，可以尝试 safetensors=True
    # safetensors=True, 
    # 如果模型需要，可能需要设置 trust_remote_code
    trust_remote_code=False 
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

print("开始量化模型...")
# .quantize() 方法执行核心的量化过程
# 它会使用一个默认的小型校准数据集 (mit-han-lab/pile-val-backup)
model.quantize(tokenizer, quant_config=quant_config)

print("量化完成！")

# --- 4. 保存量化后的模型和分词器 ---
# 保存模型权重和配置文件
# 注意：这里使用的是 model.save_quantized() 而不是 model.save_pretrained()
model.save_quantized(quant_path)
# 保存分词器，以便后续使用
tokenizer.save_pretrained(quant_path)

print(f"量化模型已保存至: {quant_path}")
print("现在你可以运行推理脚本来使用它了。")