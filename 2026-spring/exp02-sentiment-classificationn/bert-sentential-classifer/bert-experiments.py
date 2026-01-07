# bert-experiments.py
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from transformers import BertTokenizer

from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import SentimentClassifier


# ----------------- 1. 复用你原来的镜像设置 -----------------
def set_hf_mirrors():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = './hf_cache'


set_hf_mirrors()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# 固定随机种子，保证可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ----------------- 2. 计算完整指标的 evaluate_full -----------------
def evaluate_full(model, eval_loader, device):
    """
    在给定数据集上计算 Loss / Acc / F1 / AUC / 混淆矩阵
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(eval_loader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "cm": cm
    }
    return metrics


# ----------------- 3. 单次训练：保持和 main.py 基本一致 -----------------
def train_once(train_texts, train_labels,
               val_texts, val_labels,
               num_epochs,  # 这里允许我们改变 epoch 数
               run_name=""):
    """
    训练一次模型，返回在 val/test 上的完整指标
    """
    config = Config()
    config.num_epochs = num_epochs  # 覆盖轮数

    # 初始化 tokenizer 和模型（和 main.py 一样）
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.to(DEVICE)

    # 构造数据集与 DataLoader
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    val_dataset = SentimentDataset(val_texts,   val_labels,   tokenizer, config.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # 训练循环（打印风格尽量和 main.py 类似）
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f'[{run_name}] Epoch {epoch + 1}/{config.num_epochs}')
        print(f'Average training loss: {avg_train_loss:.4f}')

        # 每个 epoch 用简单 accuracy 看一眼（和原来的 evaluate 类似）
        val_metrics_epoch = evaluate_full(model, val_loader, DEVICE)
        print(f'Validation Loss: {val_metrics_epoch["loss"]:.4f}')
        print(f'Validation Accuracy: {val_metrics_epoch["accuracy"]:.4f}')

    # 训练完成后，用完整的 evaluate_full 再算一次
    final_val_metrics = evaluate_full(model, val_loader, DEVICE)
    return model, final_val_metrics


# ----------------- 4. 主实验：随机抽样 + 控制变量 -----------------
def main_experiments():
    config = Config()
    data_loader = DataLoaderClass(config)

    # 训练集先从 train.csv 读，然后构建一个“最多 2500 条的池子”
    full_train_texts, full_train_labels = data_loader.load_csv(config.train_path)
    print(f"原始训练集共 {len(full_train_texts)} 条")

    max_pool_size = min(2500, len(full_train_texts))
    # 这里先打乱再取前 max_pool_size 条，作为“样本池”
    indices_pool = np.random.permutation(len(full_train_texts))[:max_pool_size]
    train_pool_texts = [full_train_texts[i] for i in indices_pool]
    train_pool_labels = [full_train_labels[i] for i in indices_pool]
    print(f"构建训练样本池大小: {max_pool_size}")

    # 验证集和测试集不做随机抽样，固定使用 dev / test
    val_texts, val_labels = data_loader.load_csv(config.dev_path)
    test_texts, test_labels = data_loader.load_csv(config.test_path)

    # 用于保存所有结果
    results = []

    # ===== 实验 A：固定 epoch，改变训练样本量 =====
    FIXED_EPOCHS = 4
    TRAIN_SIZES = [500, 1000, 1500, 2000, 2500]

    for N in TRAIN_SIZES:
        if N > max_pool_size:
            print(f"警告: 训练样本池只有 {max_pool_size} 条，无法抽取 {N} 条，跳过该档位")
            continue

        # 从样本池里随机不放回抽 N 条
        idx = np.random.choice(max_pool_size, size=N, replace=False)
        train_texts_sub = [train_pool_texts[i] for i in idx]
        train_labels_sub = [train_pool_labels[i] for i in idx]

        run_name = f"train_size={N},epochs={FIXED_EPOCHS}"
        print("=" * 70)
        print(f"[Experiment A] {run_name}")

        # 训练 & 在验证集上评估
        model, val_metrics = train_once(
            train_texts_sub, train_labels_sub,
            val_texts, val_labels,
            num_epochs=FIXED_EPOCHS,
            run_name=run_name
        )

        # 训练完对 test 集做一次评估
        tokenizer = BertTokenizer.from_pretrained(config.model_name)
        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, config.max_seq_length)
        test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        test_metrics = evaluate_full(model, test_loader, DEVICE)

        results.append({
            "exp_type": "train_size",
            "train_size": N,
            "epochs": FIXED_EPOCHS,
            "val_loss":  val_metrics["loss"],
            "val_acc":   val_metrics["accuracy"],
            "val_f1":    val_metrics["f1"],
            "val_auc":   val_metrics["auc"],
            "test_loss": test_metrics["loss"],
            "test_acc":  test_metrics["accuracy"],
            "test_f1":   test_metrics["f1"],
            "test_auc":  test_metrics["auc"],
        })

        print("Validation confusion matrix:\n", val_metrics["cm"])
        print("Test confusion matrix:\n", test_metrics["cm"])

    # ===== 实验 B：固定训练样本量 2500，改变 epoch (1~6) =====
    FIXED_TRAIN_SIZE = min(2500, max_pool_size)
    EPOCH_LIST = [1, 2, 3, 4, 5, 6]

    for ep in EPOCH_LIST:
        # 一样，从样本池随机抽 FIXED_TRAIN_SIZE 条
        idx = np.random.choice(max_pool_size, size=FIXED_TRAIN_SIZE, replace=False)
        train_texts_sub = [train_pool_texts[i] for i in idx]
        train_labels_sub = [train_pool_labels[i] for i in idx]

        run_name = f"train_size={FIXED_TRAIN_SIZE},epochs={ep}"
        print("=" * 70)
        print(f"[Experiment B] {run_name}")

        model, val_metrics = train_once(
            train_texts_sub, train_labels_sub,
            val_texts, val_labels,
            num_epochs=ep,
            run_name=run_name
        )

        tokenizer = BertTokenizer.from_pretrained(config.model_name)
        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, config.max_seq_length)
        test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        test_metrics = evaluate_full(model, test_loader, DEVICE)

        results.append({
            "exp_type": "epochs",
            "train_size": FIXED_TRAIN_SIZE,
            "epochs": ep,
            "val_loss":  val_metrics["loss"],
            "val_acc":   val_metrics["accuracy"],
            "val_f1":    val_metrics["f1"],
            "val_auc":   val_metrics["auc"],
            "test_loss": test_metrics["loss"],
            "test_acc":  test_metrics["accuracy"],
            "test_f1":   test_metrics["f1"],
            "test_auc":  test_metrics["auc"],
        })

        print("Validation confusion matrix:\n", val_metrics["cm"])
        print("Test confusion matrix:\n", test_metrics["cm"])

    # -------- 保存成 CSV --------
    os.makedirs("exp_results", exist_ok=True)
    csv_path = os.path.join("exp_results", "bert_experiment_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print("所有实验结果已保存到:", csv_path)

    # -------- 画图：样本量 vs Dev Accuracy --------
    sub = df[df["exp_type"] == "train_size"]
    if not sub.empty:
        plt.figure()
        plt.plot(sub["train_size"], sub["val_acc"], marker="o")
        plt.title("BERT: Dev Accuracy vs Train Size")
        plt.xlabel("Train Size")
        plt.ylabel("Dev Accuracy")
        plt.grid(True)
        plt.savefig(os.path.join("exp_results",
                                 "bert_dev_acc_vs_train_size.png"),
                    dpi=300)

    # -------- 画图：Epoch vs Dev Accuracy --------
    sub = df[df["exp_type"] == "epochs"]
    if not sub.empty:
        plt.figure()
        plt.plot(sub["epochs"], sub["val_acc"], marker="o")
        plt.title("BERT: Dev Accuracy vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Dev Accuracy")
        plt.grid(True)
        plt.savefig(os.path.join("exp_results",
                                 "bert_dev_acc_vs_epochs.png"),
                    dpi=300)


if __name__ == "__main__":
    main_experiments()
