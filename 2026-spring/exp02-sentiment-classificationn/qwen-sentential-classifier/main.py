# main.py
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_hf_mirrors():
    """
    设置 Hugging Face 镜像，加速模型下载
    """
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = './hf_cache'


set_hf_mirrors()


# ================== 这里开始：直接在 main.py 里定义 SentimentClassifier ==================
class SentimentClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(SentimentClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # 使用预训练的 BertForSequenceClassification
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 返回 logits，用于交叉熵损失
        return outputs.logits

    def save_model(self, model_save_path: str):
        """
        保存模型参数到指定路径
        """
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.state_dict(), model_save_path)
        print(f"模型已保存到: {model_save_path}")

    def load_model(self, model_save_path: str, device: torch.device = None):
        """
        从指定路径加载模型参数
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_save_path, map_location=device)
        self.load_state_dict(state_dict)
        print(f"已从 {model_save_path} 加载模型参数")
# ================== 这里结束：SentimentClassifier 定义完毕 ==================


def evaluate(model, eval_loader, device):
    """
    评估模型性能
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_loader)
    accuracy = correct_predictions.double() / total_predictions
    return avg_loss, accuracy


def train(train_texts, train_labels, val_texts=None, val_labels=None):
    """
    训练模型
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if tokenizer.pad_token is None:
        # 有些 tokenizer 没有 pad_token，用 eos 代替
        tokenizer.pad_token = tokenizer.eos_token

    model = SentimentClassifier(config.model_name, config.num_classes)
    model.to(device)

    # 训练集 DataLoader
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # 验证集 DataLoader
    val_loader = None
    if val_texts is not None and val_labels is not None:
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.max_seq_length)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    total_steps = len(train_loader) * config.num_epochs

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0.0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        print(f"Epoch {epoch + 1}/{config.num_epochs}")

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        if val_loader is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                model.save_model(config.model_save_path)
                print(f"保存新的最佳模型，准确率: {val_accuracy:.4f}")

    return model


def predict(text, model_path=None):
    """
    使用训练好的模型进行预测
    """
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SentimentClassifier(config.model_name, config.num_classes)
    if model_path is not None and os.path.exists(model_path):
        model.load_model(model_path, device=device)
    else:
        print("警告: 未找到已保存模型，将使用当前未训练/未加载的模型进行预测。")

    model.to(device)
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config.max_seq_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)

    return preds.item()


if __name__ == "__main__":
    set_hf_mirrors()
    config = Config()

    data_loader = DataLoaderClass(config)

    print("加载训练集...")
    train_texts, train_labels = data_loader.load_csv(
        "C:/Users/lh008/Desktop/exp02-sentiment-classificationn/dataset/train.csv"
    )
    print("加载验证集...")
    val_texts, val_labels = data_loader.load_csv(
        "C:/Users/lh008/Desktop/exp02-sentiment-classificationn/dataset/dev.csv"
    )
    print("加载测试集...")
    test_texts, test_labels = data_loader.load_csv(
        "C:/Users/lh008/Desktop/exp02-sentiment-classificationn/dataset/test.csv"
    )

    print(f"训练集: {len(train_texts)} 样本")
    print(f"验证集: {len(val_texts)} 样本")
    print(f"测试集: {len(test_texts)} 样本")

    print("开始训练模型...")
    model = train(train_texts, train_labels, val_texts, val_labels)

    # 训练完用最佳模型预测一个示例
    example_text = "这个产品质量非常好，我很满意！"
    prediction = predict(example_text, config.model_save_path)
    sentiment = "正面" if prediction == 1 else "负面"
    print(f"示例文本: '{example_text}'")
    print(f"情感预测: {sentiment} (类别 {prediction})")
