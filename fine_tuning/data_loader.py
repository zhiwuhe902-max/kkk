import json
import os
import glob
from datasets import load_dataset, DatasetDict, Dataset
from functools import partial

def formatting_chat_func(example, tokenizer):
    """
    格式化ChatML/Qwen风格的数据集。
    (此函数保持不变，以备将来使用)
    """
    text = tokenizer.apply_chat_template(
        example['conversation'],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

def preprocess_for_alpaca(examples, tokenizer, max_length):
    """
    一个函数处理Alpaca格式的格式化、Tokenize和标签生成。
    这是更稳健的方法，可以精确控制标签。

    :param examples: huggingface datasets map函数提供的批处理数据
    :param tokenizer: 分词器实例
    :param max_length: 最大长度
    :return: Tokenized inputs including labels
    """
    # Trainer的输入应该是字典形式
    all_model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    # 遍历批处理中的每一个样本
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i] if 'input' in examples and examples['input'][i] else None
        output_text = examples['output'][i]

        # 1. 构建问题部分（Prompt）和完整文本
        if input_text and input_text.strip():
            prompt_template = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input_text}\n\n"
                "### Response:\n"
            )
            prompt = prompt_template.format(instruction=instruction, input_text=input_text)
        else:
            prompt_template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Response:\n"
            )
            prompt = prompt_template.format(instruction=instruction)
        
        # 确保output_text不为None
        output_text = output_text if output_text else ""
        full_text = prompt + output_text

        # 2. 对完整文本进行tokenize
        full_tokens = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        # 添加 EOS token (如果模型需要且tokenizer没有自动添加)
        if full_tokens["input_ids"] and full_tokens["input_ids"][-1] != tokenizer.eos_token_id:
            full_tokens["input_ids"].append(tokenizer.eos_token_id)
            full_tokens["attention_mask"].append(1)

        # 3. 单独对问题部分（Prompt）进行tokenize，以确定其长度
        prompt_tokens = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        prompt_length = len(prompt_tokens['input_ids'])

        # 4. 创建标签
        labels = list(full_tokens['input_ids'])
        
        # 将问题部分的标签设置为-100，使其不参与损失计算
        for j in range(prompt_length):
            labels[j] = -100
        
        # 确保input_ids和labels长度在截断后一致
        if len(full_tokens['input_ids']) > max_length:
            full_tokens['input_ids'] = full_tokens['input_ids'][:max_length]
            full_tokens['attention_mask'] = full_tokens['attention_mask'][:max_length]

        labels = labels[:len(full_tokens['input_ids'])]

        all_model_inputs["input_ids"].append(full_tokens['input_ids'])
        all_model_inputs["attention_mask"].append(full_tokens['attention_mask'])
        all_model_inputs["labels"].append(labels)

    return all_model_inputs


def load_and_prepare_dataset(config, tokenizer):
    """
    加载、分割和预处理数据集。
    """
    print("--- 开始加载和预处理数据 ---")
    
    dataset_config = config['dataset_config']
    train_path_str = dataset_config['train_path']
    dataset_format = dataset_config.get('format', 'alpaca')
    validation_path = dataset_config.get('validation_path', None)

    # --- 智能判断路径是文件还是文件夹 ---
    data_files = None
    if os.path.isdir(train_path_str):
        # 如果提供的是一个文件夹路径
        print(f"路径 '{train_path_str}' 是一个文件夹，正在递归查找所有 .json 文件...")
        
        # --- 核心修改在这里 ---
        # 使用 '**' 和 recursive=True 来查找所有子目录中的 .json 文件
        search_pattern = os.path.join(train_path_str, "**", "*.json")
        data_files = glob.glob(search_pattern, recursive=True)
        
        if not data_files:
            raise FileNotFoundError(f"错误: 在文件夹 '{train_path_str}' 及其所有子文件夹中没有找到任何 .json 文件。")
        print(f"找到了 {len(data_files)} 个 json 文件进行加载。")
    else:
        # 如果提供的是一个文件路径（保持原有行为）
        data_files = train_path_str
    
    print(f"使用 '{dataset_format}' 格式加载数据集...")

    # --- 数据加载 ---
    if dataset_format == 'chat':
        # (chat格式的加载逻辑，如果需要请在此处实现)
        raise NotImplementedError("Chat format loading is not fully implemented in this script version.")
    else: # alpaca 格式
        try:
            # 使用 data_files 变量加载一个或多个文件
            raw_dataset = load_dataset('json', data_files=data_files)
        except Exception as e:
            print(f"错误: 加载数据集失败: {data_files}。请检查路径和文件格式。详细信息: {e}")
            raise
    print(f"原始数据集加载完成。")

    # --- 数据集分割 ---
    if "validation" not in raw_dataset and "test" not in raw_dataset:
        if validation_path and validation_path.strip():
            print(f"加载指定的验证集: {validation_path}")
            # (此处的验证集加载逻辑可以根据需要进行扩展)
        else:
            print("未找到验证集，正在从训练集分割...")
            split_dataset = raw_dataset["train"].train_test_split(
                test_size=dataset_config['test_size'],
                seed=42
            )
            raw_dataset = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"]
            })

    # --- 核心处理步骤：一步完成格式化、Tokenize和标签制作 ---
    print("正在进行统一的数据预处理...")
    
    max_length = dataset_config['max_length']
    
    if dataset_format == 'alpaca':
        # 使用新的、稳健的函数
        preprocess_function = partial(preprocess_for_alpaca, tokenizer=tokenizer, max_length=max_length)
        
        # 定义原始数据列，以便在处理后移除
        original_columns = raw_dataset["train"].column_names
        
        tokenized_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=original_columns,
            desc="Running tokenizer on dataset",
        )
    else:
        raise ValueError(f"不支持的数据集格式: {dataset_format}")

    print("--- 数据加载和预处理完成 ---")
    return tokenized_dataset