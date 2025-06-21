from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score
import numpy as np

# 1. 加载数据并切分
dataset = load_dataset('json', data_files='../datasets/train.json', split='train')
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# 2. 加载分词器和模型
model_name = "yuchuantian/AIGC_detector_enbeta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. 数据预处理
def preprocess(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=256)

train_dataset = train_dataset.map(preprocess, batched=True)
eval_dataset = eval_dataset.map(preprocess, batched=True)

# 4. 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_steps=10,
    load_best_model_at_end=False,
    metric_for_best_model="f1",
)

# 5. F1-score指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(np.array(logits), axis=1)
    return {"f1": f1_score(np.array(labels), preds)}

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# 7. 训练与评估
trainer.train()
results = trainer.evaluate()

trainer.save_model('./results')