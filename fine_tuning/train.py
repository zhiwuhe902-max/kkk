# fine_tuning/train.py

print("--- Python 脚本开始执行 ---")

import os
import yaml
import torch
from functools import partial
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig
# 如果未来想支持更多方法，在这里或下面映射中添加即可
# from peft import AdaLoraConfig, PrefixTuningConfig


from data_loader import load_and_prepare_dataset
from metrics import compute_metrics

# PEFT 配置映射: 将 config.yaml 中的字符串映射到实际的 PEFT 配置类
PEFT_CONFIG_MAPPING = {
    "lora": LoraConfig,
}

def main():
    print("--- 步骤 1/9: 正在加载配置 ---")
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置加载成功。")
    except FileNotFoundError as e:
        print(f"错误: config.yaml 文件未找到。请确保配置文件存在于当前目录。详细信息: {e}")
        return
    except Exception as e:
        print(f"错误: 解析 config.yaml 文件时出错。请检查文件格式是否正确。详细信息: {e}")
        return

    model_config = config['model_config']
    quant_config = config.get('quantization_config', {})
    peft_config = config.get('peft_config', {})
    training_args_params = config['training_args']
    viz_config = config['visualization']

    print("\n--- 步骤 2/9: 正在设置 Weights & Biases ---")
    # 检查是否需要离线模式
    if os.environ.get("WANDB_MODE") != "offline":
        try:
            import wandb
            wandb.login()
            os.environ["WANDB_PROJECT"] = viz_config['wandb_project']
            report_to = ["tensorboard", "wandb"]
            print(f"Weights & Biases 已启用。项目: {viz_config['wandb_project']}")
        except ImportError:
            print("警告: 'wandb' 未安装。将仅使用 TensorBoard。运行 'pip install wandb' 来启用。")
            report_to = "tensorboard"
        except Exception as e:
            print(f"W&B 登录失败，将以离线模式运行。错误: {e}")
            os.environ["WANDB_MODE"] = "offline"
            report_to = ["tensorboard", "wandb"]
    else:
        print("W&B 已设置为离线模式。")
        report_to = ["tensorboard", "wandb"]

    print("\n--- 步骤 3/9: 正在加载分词器 ---")
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
    tokenizer.pad_token = tokenizer.eos_token
    print("分词器加载完成。")

    print("\n--- 步骤 4/9: 正在配置量化 ---")
    bnb_config = None
    if quant_config.get('enabled', False):
        print("量化已启用 (BitsAndBytes)。")
        # 提取 BitsAndBytesConfig 的参数
        bnb_params = {k: v for k, v in quant_config.items() if k != 'enabled'}

        # 确保 compute_dtype 是 torch.dtype 类型
        if 'bnb_4bit_compute_dtype' in bnb_params:
            compute_dtype_str = bnb_params.pop('bnb_4bit_compute_dtype')
            if compute_dtype_str == 'bfloat16':
                bnb_params['bnb_4bit_compute_dtype'] = torch.bfloat16
            elif compute_dtype_str == 'float16':
                bnb_params['bnb_4bit_compute_dtype'] = torch.float16

        bnb_config = BitsAndBytesConfig(**bnb_params)
        print(f"BitsAndBytes 配置: {bnb_config.to_dict()}")
    else:
        print("量化未启用。")

    print("\n--- 步骤 5/9: 正在加载模型 ---")
    # 根据是否量化来确定 torch_dtype
    model_dtype = None
    if not bnb_config:
        # 仅在非量化时从config加载dtype
        model_dtype = torch.float16 if model_config['torch_dtype'] == 'float16' else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        device_map="auto",
        torch_dtype=model_dtype,
        quantization_config=bnb_config
    )
    print("模型加载完成。")

    print("\n--- 步骤 6/9: 正在配置微调方法 ---")
    if peft_config.get('enabled', False):
        print("PEFT 微调已启用。")

        # 关键修复: 为 PEFT + 梯度检查点 启用梯度输入
        # 这可以防止在冻结大部分模型层时梯度计算中断
        model.enable_input_require_grads()

        peft_method = peft_config.get('method')
        peft_params = peft_config.get('params')

        if not peft_method or not peft_params:
            print("错误: 'peft_config.enabled' 为 true, 但 'method' 或 'params' 未在 config.yaml 中配置。")
            return

        print(f"正在应用 PEFT 方法: {peft_method.upper()}")
        peft_config_class = PEFT_CONFIG_MAPPING.get(peft_method.lower())
        if not peft_config_class:
            print(f"错误: 不支持的 PEFT 方法 '{peft_method}'. 支持的方法: {list(PEFT_CONFIG_MAPPING.keys())}")
            return

        # 动态创建 PEFT 配置
        peft_task_config = peft_config_class(**peft_params)

        model = get_peft_model(model, peft_task_config)
        print(f"{peft_method.upper()} 适配器已应用。可训练参数:")
        model.print_trainable_parameters()
    else:
        print("全参数微调已启用。")

    print("\n--- 步骤 7/9: 正在加载和处理数据 ---")
    tokenized_dataset = load_and_prepare_dataset(config, tokenizer)
    print("数据加载和处理完成。")
    print(f"训练集样本数: {len(tokenized_dataset['train'])}")
    print(f"评估集样本数: {len(tokenized_dataset['validation'])}")


    print("\n--- 步骤 8/9: 正在配置训练器 ---")
    training_args = TrainingArguments(
        **training_args_params,
        report_to=report_to,
    )

    # 使用 functools.partial 传递 tokenizer 到 compute_metrics
    compute_metrics_with_tokenizer = partial(compute_metrics, tokenizer=tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_tokenizer,
    )
    print("训练器配置完成。")

    print("\n--- 步骤 9/9: 开始训练 ---")
    trainer.train()
    print("--- 训练完成 ---")

    # --- 保存模型 ---
    print("\n--- 正在保存最终模型 ---")
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n--- 模型和分词器已成功保存至: {final_model_path} ---")

if __name__ == "__main__":
    main()