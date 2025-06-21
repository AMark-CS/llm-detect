import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# 1. 加载数据
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl('../datasets/train.json')
test_data = load_jsonl('../datasets/test.json')

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# 2. Tokenizer和模型
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. LoRA配置
lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)

# 4. 数据预处理
def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

# 5. 评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# 6. 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

# 7. Trainer
split = train_dataset.train_test_split(test_size=0.1, seed=42)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# 8. 训练
trainer.train()

# 9. 绘制loss和F1曲线
log_history = trainer.state.log_history
train_losses = [log['loss'] for log in log_history if 'loss' in log and 'epoch' in log]
eval_f1s = [log['eval_f1'] for log in log_history if 'eval_f1' in log]
epochs = [log['epoch'] for log in log_history if 'loss' in log and 'epoch' in log]
eval_epochs = [log['epoch'] for log in log_history if 'eval_f1' in log]

plt.figure(figsize=(10,5))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(eval_epochs, eval_f1s, label='Validation F1', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Epoch vs Loss/F1')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('epoch_loss_f1.png')
plt.close()

# 10. 推理并保存到 submit.txt
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(-1)

with open("submit.txt", "w") as f:
    for label in pred_labels:
        f.write(f"{label}\n")

print("训练与推理完成，loss/F1曲线已保存为 epoch_loss_f1.png，预测结果已保存到 submit.txt")