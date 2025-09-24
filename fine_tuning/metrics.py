# fine_tuning/metrics.py

import numpy as np
import evaluate
import torch

rouge = evaluate.load("./rouge")

def compute_metrics(eval_preds, tokenizer):
    """
    计算评估指标，包括 ROUGE 和 Perplexity。

    :param eval_preds: Trainer返回的EvalPrediction对象
    :param tokenizer: Hugging Face Tokenizer 实例
    :return: 包含指标的字典
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # 将-100替换为pad_token_id，以便解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # 解码预测和标签
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE 分数
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Perplexity
    try:
        # 从logits计算loss
        logits = torch.from_numpy(preds)
        labels_tensor = torch.from_numpy(labels)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels_tensor[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss).item()
        result["perplexity"] = perplexity
    except Exception as e:
        print(f"计算Perplexity时出错: {e}")
        result["perplexity"] = -1

    return result