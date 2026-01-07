# qwen-experiments.py

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass

# ---------------- 0. 基础设置 ----------------
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_hf_mirrors():
    """
    和 main.py 一样的镜像配置
    """
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = './hf_cache'
    # 如果你确认本地已有模型又老是连不上，可以打开离线：
    # os.environ['HF_HUB_OFFLINE'] = '1'


set_hf_mirrors()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------- 1. 完整评估函数：Loss / Acc / F1 / AUC / 混淆矩阵 ----------------
def evaluate_full(model, eval_loader, device):
    """
    评估完整指标：Loss / Accuracy / F1 / AUC / 混淆矩阵
    model 必须是 AutoModelForSequenceClassification 或兼容接口
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            loss = loss_fn(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

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


# ---------------- 2. 单次训练：结构尽量贴近你原来的 train() ----------------
def train_once(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    num_epochs: int,
    run_name: str = ""
):
    """
    训练一次 Qwen 模型（或者 Config 里指定的模型），返回验证集完整指标
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = Config()
    # 覆盖 epoch 数，用于做 epoch 控制变量实验
    config.num_epochs = num_epochs

    device = DEVICE
    print(f"[{run_name}] 使用设备: {device}")

    # tokenizer 与 main.py 一致
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # backbone 与 main.py 的 SentimentClassifier 内部一致
    backbone = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_classes
    ).to(device)

    # Dataset / DataLoader
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    val_dataset   = SentimentDataset(val_texts,   val_labels,   tokenizer, config.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False)

    total_steps = len(train_loader) * config.num_epochs

    optimizer = AdamW(backbone.parameters(), lr=config.learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
        backbone.train()
        total_loss = 0.0

        print(f"[{run_name}] Epoch {epoch + 1}/{config.num_epochs}")

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[{run_name}] Average training loss: {avg_train_loss:.4f}")

        # 每个 epoch 看一眼验证集完整指标
        val_metrics_epoch = evaluate_full(backbone, val_loader, device)
        print(f"[{run_name}] Validation Loss: {val_metrics_epoch['loss']:.4f}")
        print(f"[{run_name}] Validation Accuracy: {val_metrics_epoch['accuracy']:.4f}")

    # 训练结束后，返回最终在验证集上的指标
    final_val_metrics = evaluate_full(backbone, val_loader, device)
    return backbone, tokenizer, final_val_metrics


# ---------------- 3. 主实验：随机抽样 + 控制变量 ----------------
def main_experiments():
    config = Config()
    data_loader = DataLoaderClass(config)

    # 1) 统一在这里根据当前脚本位置拼 dataset 绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 当前 qwen-experiments.py 所在目录
    train_path = os.path.join(base_dir, "..", "dataset", "train.csv")
    dev_path   = os.path.join(base_dir, "..", "dataset", "dev.csv")
    test_path  = os.path.join(base_dir, "..", "dataset", "test.csv")

    print("加载训练集...")
    full_train_texts, full_train_labels = data_loader.load_csv(train_path)
    print("加载验证集...")
    val_texts, val_labels = data_loader.load_csv(dev_path)
    print("加载测试集...")
    test_texts, test_labels = data_loader.load_csv(test_path)

    print(f"原始训练集: {len(full_train_texts)} 样本")

    # 2) 构建“样本池”：最多 2500 条（先打乱再取前 2500）
    max_pool_size = min(2500, len(full_train_texts))
    indices_pool = np.random.permutation(len(full_train_texts))[:max_pool_size]
    train_pool_texts = [full_train_texts[i] for i in indices_pool]
    train_pool_labels = [full_train_labels[i] for i in indices_pool]
    print(f"训练样本池大小: {max_pool_size}")

    results = []

    # ========== 实验 A：固定 epoch=4，改变训练样本量 ==========
    FIXED_EPOCHS = 4
    TRAIN_SIZES = [500, 1000, 1500, 2000, 2500]

    for N in TRAIN_SIZES:
        if N > max_pool_size:
            print(f"样本池只有 {max_pool_size} 条，无法抽取 {N} 条，跳过该档位")
            continue

        # 从样本池随机不放回抽 N 条
        idx = np.random.choice(max_pool_size, size=N, replace=False)
        train_texts_sub = [train_pool_texts[i] for i in idx]
        train_labels_sub = [train_pool_labels[i] for i in idx]

        run_name = f"train_size={N},epochs={FIXED_EPOCHS}"
        print("=" * 70)
        print(f"[Experiment A] {run_name}")

        model, tokenizer, val_metrics = train_once(
            train_texts_sub,
            train_labels_sub,
            val_texts,
            val_labels,
            num_epochs=FIXED_EPOCHS,
            run_name=run_name
        )

        # 测试集评估
        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, config.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
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

    # ========== 实验 B：固定训练样本量 = 样本池大小，改变 epoch(1~6) ==========
    FIXED_TRAIN_SIZE = max_pool_size  # 最多能用多少就用多少
    EPOCH_LIST = [1, 2, 3, 4, 5, 6]

    for ep in EPOCH_LIST:
        if FIXED_TRAIN_SIZE == 0:
            print("样本池为空，无法进行 epoch 实验")
            break

        idx = np.random.choice(max_pool_size, size=FIXED_TRAIN_SIZE, replace=False)
        train_texts_sub = [train_pool_texts[i] for i in idx]
        train_labels_sub = [train_pool_labels[i] for i in idx]

        run_name = f"train_size={FIXED_TRAIN_SIZE},epochs={ep}"
        print("=" * 70)
        print(f"[Experiment B] {run_name}")

        model, tokenizer, val_metrics = train_once(
            train_texts_sub,
            train_labels_sub,
            val_texts,
            val_labels,
            num_epochs=ep,
            run_name=run_name
        )

        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, config.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
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

    # ========== 保存结果 & 画图 ==========
    os.makedirs("exp_results", exist_ok=True)
    csv_path = os.path.join("exp_results", "qwen_experiment_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print("所有 Qwen 实验结果已保存到:", csv_path)

    # 样本量 vs Dev Accuracy
    sub = df[df["exp_type"] == "train_size"]
    if not sub.empty:
        plt.figure()
        plt.plot(sub["train_size"], sub["val_acc"], marker="o")
        plt.title("Qwen: Dev Accuracy vs Train Size")
        plt.xlabel("Train Size")
        plt.ylabel("Dev Accuracy")
        plt.grid(True)
        plt.savefig(os.path.join("exp_results",
                                 "qwen_dev_acc_vs_train_size.png"),
                    dpi=300)

    # Epoch vs Dev Accuracy
    sub = df[df["exp_type"] == "epochs"]
    if not sub.empty:
        plt.figure()
        plt.plot(sub["epochs"], sub["val_acc"], marker="o")
        plt.title("Qwen: Dev Accuracy vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Dev Accuracy")
        plt.grid(True)
        plt.savefig(os.path.join("exp_results",
                                 "qwen_dev_acc_vs_epochs.png"),
                    dpi=300)


if __name__ == "__main__":
    print(">>> qwen_experiments main_experiments() 开始执行")
    main_experiments()
    print(">>> qwen_experiments main_experiments() 执行结束")
