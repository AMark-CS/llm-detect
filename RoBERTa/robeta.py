import torch
import json
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score

# 1. 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta", num_labels=2)

# 2. 构建数据集
class TextDetectDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=256):
        self.samples = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                label = 1 if item['label'] == 1 else 0
                self.samples.append({'text': item['text'], 'label': label})
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]['text']
        label = self.samples[idx]['label']
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# 3. 加载数据集并划分（7:3）
dataset = TextDetectDataset('./train.json', tokenizer)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 4. 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='./logs',
)

# 5. F1评价指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    f1 = f1_score(labels, preds)
    return {"f1": f1}

# 6. 训练与评估
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_result = trainer.evaluate()
print("Test set evaluation:", eval_result)