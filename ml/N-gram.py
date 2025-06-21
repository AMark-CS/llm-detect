import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 1. 读取数据
texts, labels = [], []
with open("../data/train.json", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])
        labels.append(obj["label"])

# 2. 切分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.37, random_state=42)

# 3. N-gram 特征（如1-3gram）
vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=5000)
X_train_ngram = vectorizer.fit_transform(X_train)
X_val_ngram = vectorizer.transform(X_val)

# 4. 训练分类器
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_ngram, y_train)

# 5. 验证集评估
y_pred = clf.predict(X_val_ngram)
print("F1-score:", f1_score(y_val, y_pred))

# 6. 用于test.json预测并导出submit.txt
test_texts = []
with open("../data/test.json", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        test_texts.append(obj["text"])

X_test_ngram = vectorizer.transform(test_texts)
test_pred = clf.predict(X_test_ngram)

with open("../data/submit.txt", "w", encoding="utf-8") as fout:
    for label in test_pred:
        fout.write(str(label) + "\n")