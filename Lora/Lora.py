import pandas as pd
import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from datetime import datetime
import matplotlib.pyplot as plt


# 1. 数据加载与预处理
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# 加载训练数据
train_df = load_data("../datasets/train.json")
test_df = load_data("../datasets/test.json")

# 转换为HuggingFace Dataset格式
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 修改后的关键代码段如下：
# 模型与tokenizer加载部分
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充token
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id  # 同步配置

# 3. LoRA配置
lora_config = LoraConfig(
    r=256,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)

# 4. 数据预处理函数

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        pad_to_max_length=True,
        return_special_tokens_mask=True
    )

# 处理数据集
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# model.requires_grad_(False)  # 冻结主模型
# model.get_adapter("default").requires_grad_(True)  # 仅训练LoRA


# 5. 定义评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 6. 训练参数配置

# 修改训练参数（添加早停）
# 修改后的训练参数
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,  # 降低学习率
    per_device_train_batch_size=16,
    num_train_epochs=4,  # 增加训练轮数
    weight_decay=0.05,    # 强化正则化
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    disable_tqdm=False,
    push_to_hub=False,
    label_smoothing_factor=0.1,  # 标签平滑
    lr_scheduler_type="cosine_with_restarts",
)
# 将原始训练集划分为训练集和验证集（80%训练 + 20%验证）
train_subset, val_subset = train_dataset.train_test_split(test_size=0.2).values()

# Trainer初始化
# 修改Trainer的eval_dataset参数
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=val_subset,  # 使用划分出的验证集
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    # 设置损失函数早停，增强模型泛化能力
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 8. 模型训练
trainer.train()

# 9. 模型评估
# 最终评估时使用独立测试集


# 提取训练损失（每个epoch的平均值）
train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
epochs = list(range(1, len(train_losses) + 1))

# 提取验证集F1分数
eval_metrics = [log for log in trainer.state.log_history if 'eval_f1' in log]
eval_epochs = [metric['epoch'] for metric in eval_metrics]
eval_f1s = [metric['eval_f1'] for metric in eval_metrics]

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
plt.plot(eval_epochs, eval_f1s, label='Validation F1', color='orange', marker='s')

# 设置双Y轴
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(eval_epochs, eval_f1s, visible=False)  # 创建隐式轴

# 添加标题和标签
plt.title('Training Loss vs Validation F1 Over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss', color='blue')
ax2.set_ylabel('Validation F1', color='orange')

# 图例合并显示
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.grid(True)
plt.tight_layout()
plt.savefig(f"loss_vs_f1_{datetime.now().strftime('%Y%m%d_%H%M')}.png")


def plot_metrics(log_history):
    # 分离训练和验证记录
    train_losses = [log['loss'] for log in log_history if 'loss' in log and 'eval_loss' not in log]
    eval_metrics = [log for log in log_history if 'eval_loss' in log]

    epochs = list(range(1, len(train_losses) + 1))
    eval_epochs = [m['epoch'] for m in eval_metrics]

    # 创建画布
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 主Y轴：Loss
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, 'o-', color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # 次Y轴：F1/Accuracy
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Metrics', color=color)
    ax2.plot(eval_epochs, [m['eval_f1'] for m in eval_metrics], 's-', color=color, label='F1')
    ax2.plot(eval_epochs, [m['eval_accuracy'] for m in eval_metrics], 'd-', color='green', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加其他指标
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))  # 移动第三个轴
    color = 'tab:red'
    ax3.set_ylabel('Precision/Recall', color=color)
    ax3.plot(eval_epochs, [m['eval_precision'] for m in eval_metrics], '^-', color=color, label='Precision')
    ax3.plot(eval_epochs, [m['eval_recall'] for m in eval_metrics], 'v-', color='purple', label='Recall')
    ax3.tick_params(axis='y', labelcolor=color)

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               ncol=3)

    plt.title('Multi-Metric Visualization')
    fig.tight_layout()

    plt.savefig(f"metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.png")


plot_metrics( trainer.state.log_history)

# 10. 预测并保存结果
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(-1)

with open("submit.txt", "w") as f:
    for label in pred_labels:
        f.write(f"{label}\n")
