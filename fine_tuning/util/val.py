import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer # 用于加载分词器和应用聊天模板

# --- 1. 配置参数 ---
MODEL_PATH = '/home/Qwen3-7B' # 你的模型路径
INPUT_JSON_PATH = '/home/ly/LLMs/alpaca_zh.json' # 你的问题文件路径
OUTPUT_JSON_PATH = '/home/ly/LLMs/output.json' # 输出文件路径
TENSOR_PARALLEL_SIZE = 2
GPU_MEMORY_UTILIZATION = 0.8

# (可选) 为所有生成任务设置统一的系统提示
# 如果你的 questions.json 中的 "instruction" 已经是完整的用户指令，你可能不需要额外的系统提示
# 或者你可以根据需要调整
# DEFAULT_SYSTEM_PROMPT = "你是一个乐于助人、精准且专业的AI助手。"
DEFAULT_SYSTEM_PROMPT = None # 如果不想使用统一的系统提示，设为 None

# (可选) 配置 SamplingParams
# 这些参数会影响模型的生成行为，对于评估，你可能希望它们保持一致
# 例如，为了更确定的输出，可以使用较低的 temperature
# 为了限制输出长度，可以设置 max_tokens
sampling_params = SamplingParams(
    temperature=0.7,  # 控制随机性，0 表示确定性
    top_p=0.95,       # 控制核心采样的概率阈值
    max_tokens=512,  # 生成的最大 token 数量，根据你的任务调整
    # stop=["<|im_end|>"], # 模型应该会自动使用 EOS token 停止，但也可以显式指定
)


# --- 2. 加载分词器 ---
print("正在加载分词器...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
except Exception as e:
    print(f"加载分词器失败: {e}")
    exit()

# --- 3. 加载模型 ---
print("正在加载 LLM...")
# 注意：在 vLLM 0.3.0 及之后版本，一些模型如 Qwen1.5 可能需要 `enforce_eager=True`
# 对于 Qwen3，根据你的 vLLM 版本和模型具体情况，可能需要此参数
# 如果遇到 "CUDA error: an illegal memory access was encountered" 或类似问题，可以尝试添加
# llm = LLM(
#     model=MODEL_PATH,
#     tensor_parallel_size=TENSOR_PARALLEL_SIZE,
#     gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
#     enforce_eager=True # 仅在需要时添加
# )
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION
)
print("LLM 加载完成。")

# --- 4. 加载问题数据 ---
print(f"正在从 {INPUT_JSON_PATH} 加载问题...")
try:
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    if not isinstance(questions_data, list):
        print(f"错误: {INPUT_JSON_PATH} 的内容不是一个 JSON 列表。请检查文件格式。")
        exit()
    print(f"成功加载 {len(questions_data)} 个问题。")
except FileNotFoundError:
    print(f"错误: 输入文件 {INPUT_JSON_PATH} 未找到。")
    exit()
except json.JSONDecodeError:
    print(f"错误: 输入文件 {INPUT_JSON_PATH} 不是有效的 JSON 格式。")
    exit()

# --- 5. 准备 Prompts ---
prompts_for_llm = []
processed_data_for_output = [] # 用于存储原始数据和后续添加模型输出

print("正在准备 prompts...")
for i, item in enumerate(questions_data):
    if not isinstance(item, dict):
        print(f"警告: 跳过问题 #{i+1}，因为它不是一个字典格式。")
        continue

    instruction = item.get("instruction")
    item_input = item.get("input", "") # "input" 字段可能为空

    if not instruction:
        print(f"警告: 问题 #{i+1} 缺少 'instruction' 字段，将跳过。")
        continue

    # 构建 messages 列表供 tokenizer.apply_chat_template 使用
    messages = []
    if DEFAULT_SYSTEM_PROMPT:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    # 组合 instruction 和 input 作为 user 的内容
    user_content = instruction
    if item_input and item_input.strip(): # 如果 input 不为空
        user_content += "\n" + item_input # 可以根据需要调整 instruction 和 input 的组合方式

    messages.append({"role": "user", "content": user_content})

    # 应用聊天模板
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 重要：提示模型开始生成
        )
        prompts_for_llm.append(prompt_text)
        # 保存原始数据，之后会把模型输出加进去
        processed_data_for_output.append({
            "instruction": instruction,
            "input": item_input,
            "ground_truth_output": item.get("output", "") # 保留原始的 "output" 作为参考
        })
    except Exception as e:
        print(f"为问题 #{i+1} (Instruction: {instruction[:50]}...) 创建 prompt 失败: {e}")
        # 如果某个 prompt 创建失败，也需要从 processed_data_for_output 中移除对应项，或标记错误
        # 这里简单跳过，但更健壮的做法是记录错误
        continue


if not prompts_for_llm:
    print("没有可供推理的有效 prompts。程序退出。")
    exit()

print(f"成功准备 {len(prompts_for_llm)} 个 prompts。")

# --- 6. 执行批量推理 ---
print("开始批量推理...")
# vLLM 的 generate 方法可以直接处理 prompts 列表
request_outputs = llm.generate(prompts_for_llm, sampling_params)
print("批量推理完成。")

# --- 7. 处理结果并保存到 JSON ---
results_to_save = []
if len(request_outputs) != len(processed_data_for_output):
    print("警告: 生成的输出数量与处理的问题数量不匹配。结果可能不准确。")
    # 这种情况理论上不应该发生，除非在准备 prompts 阶段有未正确处理的跳过

for i, req_output in enumerate(request_outputs):
    if i < len(processed_data_for_output):
        original_data = processed_data_for_output[i]
        model_generated_text = req_output.outputs[0].text.strip() # 获取生成的文本并去除首尾空白

        results_to_save.append({
            "instruction": original_data["instruction"],
            "input": original_data["input"],
            "ground_truth_output": original_data["ground_truth_output"], # 参考答案
            "model_output": model_generated_text # 模型生成的答案
        })
    else:
        # 理论上不应到达这里，但作为安全措施
        print(f"警告: 第 {i+1} 个推理结果没有对应的原始数据。")


print(f"正在将 {len(results_to_save)} 条结果保存到 {OUTPUT_JSON_PATH}...")
try:
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)
    print(f"结果已成功保存到 {OUTPUT_JSON_PATH}")
except Exception as e:
    print(f"保存结果到文件失败: {e}")