# Qwen LoRA 微调项目

这是一个用于对基于Transformer的大型语言模型（如Qwen系列）进行LoRA（Low-Rank Adaptation）微调的结构化项目。

## 特性

- **配置文件驱动**: 通过 `config.yaml` 轻松管理模型、数据集和训练参数。
- **模块化设计**: 数据处理、评估指标和训练逻辑相互分离，易于维护和扩展。
- **强大的可视化与日志**: 集成 TensorBoard 和 Weights & Biases (W&B)，用于实时监控训练过程（学习率、损失、评估指标、GPU使用情况等）。
- **支持离线训练**: 支持 W&B 离线模式，日志可以后续上传。
- **自动化评估**: 在评估过程中自动计算 ROUGE 和 Perplexity。

## 项目结构

```
fine_tuning/
├── config.yaml         # 主配置文件
├── data_loader.py      # 数据加载和预处理
├── metrics.py          # 评估指标计算
├── train.py            # 主训练脚本
├── requirements.txt    # 项目依赖
└── README.md           # 本文档
```

## 安装与设置

1.  **克隆项目并进入目录**
    ```bash
    # 假设您已将这些文件保存在 fine_tuning 目录中
    cd fine_tuning
    ```

2.  **创建虚拟环境 (推荐)**
    ```bash
    conda create -n FT python=3.12
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置 W&B (可选)**
    如果您想使用 Weights & Biases 进行在线监控，请登录：
    ```bash
    wandb login
    ```
    如果您想离线运行，可以在 `train.py` 中设置环境变量 `WANDB_MODE=offline`，或者修改代码逻辑。

## 如何使用

1.  **修改配置文件 `config.yaml`**
    - `model_config.name`: 设置您想微调的基础模型。
    - `dataset_config.path`: 设置您的数据集路径（支持Hugging Face Hub或本地文件）。
    - `training_args`: 根据您的硬件和需求调整训练参数，如批大小、学习率等。
    - `visualization.wandb_project`: 为您的W&B项目命名。

2.  **开始训练**
    ```bash
    python train.py
    ```

3.  **查看结果**
    - **TensorBoard**: 训练过程中，日志会保存在 `training_args.output_dir` 下的 `runs` 文件夹中。您可以使用以下命令启动TensorBoard：
      ```bash
      tensorboard --logdir=./qwen_lora_finetuned
      ```
    - **Weights & Biases**: 如果在线，可以直接访问您的W&B项目页面。如果离线，训练结束后，根据提示使用 `wandb sync` 命令上传日志。
    - **模型文件**: 训练完成后，最终的模型和分词器将保存在 `training_args.output_dir` 下的 `final_model` 文件夹中。

## 注意

- **F1-Score**: 对于生成任务，F1-Score不是一个标准指标。ROUGE分数中的F1值（例如 `rougeL_fmeasure`）是更常用的评估指标，本项目已包含。
- **GPU使用情况**: 当 `wandb` 启用时，它会自动监控并记录GPU的使用情况（如利用率、显存占用等），无需额外代码。