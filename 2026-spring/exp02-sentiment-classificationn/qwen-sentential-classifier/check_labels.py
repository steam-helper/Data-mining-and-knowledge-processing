# check_labels.py
import pandas as pd
import torch
from transformers import AutoTokenizer

from main import SentimentClassifier, Config  # 复用你 main.py 里的类


def show_some_samples(csv_path, n=10):
    """
    打印前 n 条样本，看一下 label 大概对应什么语气
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["label", "title", "text"]
    )
    df = df.fillna("")

    print("===== 前几条原始样本（只看 label 和前 50 个字）=====")
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        label = row["label"]
        title = str(row["title"])[:30]
        text = str(row["text"])[:50]
        print(f"[样本 {i}] label={label} | title={title} | text={text}")
    print("===============================================")


def check_with_model(csv_path, model_path, n=10):
    """
    用训练好的模型在前 n 条样本上跑预测，对比 label 和预测
    """
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 初始化并加载已训练模型
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.load_model(model_path, device=device)
    model.to(device)
    model.eval()

    # 读取数据
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["label", "title", "text"]
    )
    df = df.fillna("")

    print("\n===== 数据集 label vs 模型预测（前几条）=====")
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        true_label = int(row["label"])
        title = str(row["title"])
        text = str(row["text"])
        full_text = title + "。" + text

        encoding = tokenizer.encode_plus(
            full_text,
            add_special_tokens=True,
            max_length=config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            pred_label = int(preds.item())

        # 这里先不乱定义正负，只打印 label 数字
        print(f"[样本 {i}] true={true_label}, pred={pred_label}, 文本前50字: {full_text[:50]}")

    print("=======================================")


if __name__ == "__main__":
    # 根据你现在的路径来
    csv_path = r"C:\Users\lh008\Desktop\exp02-sentiment-classificationn\dataset\train.csv"
    model_path = r"C:\Users\lh008\Desktop\exp02-sentiment-classificationn\qwen-sentential-classifier\saved_models\sentiment_model.pth"

    # 1) 先看看原始 label 长什么样
    show_some_samples(csv_path, n=10)

    # 2) 再用训练好的模型对这些样本做预测
    check_with_model(csv_path, model_path, n=10)
