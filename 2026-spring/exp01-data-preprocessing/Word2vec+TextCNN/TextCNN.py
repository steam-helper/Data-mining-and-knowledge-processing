import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer  # 复用Keras的Tokenizer简化序列转换
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 确保nltk分词数据已下载
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt', quiet=True)


# --------------------------
# 1. 数据预处理（与word2vec.py完全一致，保证格式匹配）
# --------------------------
def preprocess_text(text):
    """文本清洗+分词（和word2vec.py逻辑完全相同）"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()  # 转小写
    text = re.sub(r'[^\w\s]', '', text)  # 移除特殊字符
    tokens = word_tokenize(text)  # 分词
    return tokens


def load_data(file_path):
    """加载数据集并预处理"""
    # 读取数据（与word2vec.py用相同的列名和格式）
    df = pd.read_csv(file_path, names=['label', 'title', 'review'], header=None,nrows=10000)       # 直接读取前1000条数据
    df = df.fillna('')  # 填充缺失值
    df['text'] = df['title'] + ' ' + df['review']  # 合并标题和评论

    # 生成预处理后的分词语料
    corpus = [preprocess_text(text) for text in tqdm(df['text'], desc="预处理文本")]

    # 标签转换：1→0（负面），2→1（正面）
    df['label'] = df['label'].map({1: 0, 2: 1})
    labels = df['label'].values

    return corpus, labels


# --------------------------
# 2. 自定义数据集（PyTorch标准格式）
# --------------------------
class SentimentDataset(Dataset):
    def __init__(self, corpus, labels, tokenizer, max_seq_len):
        self.corpus = corpus  # 分词语料
        self.labels = labels  # 标签（0/1）
        self.tokenizer = tokenizer  # 词→索引映射器
        self.max_seq_len = max_seq_len  # 固定序列长度

    def __len__(self):
        return len(self.corpus)  # 数据集总条数

    def __getitem__(self, idx):
        """获取单条数据，转换为模型可输入的Tensor"""
        tokens = self.corpus[idx]
        # 分词转索引序列（OOV词用<OOV>的索引）
        seq = self.tokenizer.texts_to_sequences([tokens])[0]
        # 固定序列长度（截断或填充）
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        else:
            seq += [0] * (self.max_seq_len - len(seq))  # 0为padding索引
        # 转换为Tensor
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


# --------------------------
# 3. TextCNN模型（PyTorch实现）
# --------------------------
class TextCNN(nn.Module):
    """输入：批量迭代器（每次返回 32 条样本，含 “固定长度数字序列张量 + 标签张量”），相当于只是每个词对应数字而还没有牵扯词向量嵌入
    输出：空的 TextCNN 模型（组件已搭好，权重待初始化 / 训练）；
    用途：搭建模型的 “骨架”，后续填权重、做计算；"""
    def __init__(self, vocab_size, embed_dim, w2v_model_path, tokenizer, num_filters=128):
        super(TextCNN, self).__init__()
        self.tokenizer = tokenizer  # 初始化tokenizer属性
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
            )
        self._load_word2vec_embeddings(w2v_model_path)  # 此时self.tokenizer已存在

        # 多尺寸卷积层（3、5窗口）
        #2个卷积层（conv1/k=3、conv2/k=5）：用来提取文本不同长度的局部语义（比如3个词的短语、5个词的短语）
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=5)

        # 全局最大池化 压缩卷积特征，保留核心信息，减少参数
        self.pool = nn.AdaptiveMaxPool1d(1)

        # 全连接层（分类）
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 2, 128),  # 拼接2种卷积的输出
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # 二分类输出
        )

        # 输出激活函数
        self.sigmoid = nn.Sigmoid()

    def _load_word2vec_embeddings(self, w2v_model_path):
        """用同目录的Word2Vec模型初始化嵌入层权重"""
        w2v_model = Word2Vec.load(w2v_model_path)
        oov_count = 0  # 未在Word2Vec中出现的词数量
        # 遍历词汇表，替换已知词的向量
        for word, idx in self.tokenizer.word_index.items():
            if word in w2v_model.wv:
                self.embedding.weight.data[idx] = torch.tensor(w2v_model.wv[word], dtype=torch.float32)#把 Word2Vec 中该词的向量（numpy 数组）转成 PyTorch 张量，赋值给嵌入层权重矩阵的第idx行（替换随机初始值）
                #TextCNN的嵌入层是个矩阵（形状：[词汇表大小, 向量维度]），每一行对应一个索引的向量。
            else:
                oov_count += 1
        # 打印嵌入层信息
        vocab_size = len(self.tokenizer.word_index) + 1
        print(
            f"嵌入层初始化完成 | 词汇表大小: {vocab_size} | 未识别词(OOV): {oov_count} (占比: {oov_count / vocab_size:.2%})")

    def forward(self, x):
        """前向传播：输入序列→嵌入→卷积→池化→分类
        PyTorch 在执行model(seq)时自动触发"""
        # 嵌入层：(batch_size, max_seq_len) → (batch_size, max_seq_len, embed_dim)
        x_embed = self.embedding(x)

        # 调整维度适配Conv1d：(batch_size, embed_dim, max_seq_len)
        x_embed = x_embed.permute(0, 2, 1)

        # 卷积+池化
        x1 = self.pool(torch.relu(self.conv1(x_embed))).squeeze(-1)  # (batch_size, num_filters)
        x2 = self.pool(torch.relu(self.conv2(x_embed))).squeeze(-1)  # (batch_size, num_filters)

        # 拼接特征
        x_concat = torch.cat([x1, x2], dim=1)  # (batch_size, 2*num_filters)

        # 分类输出
        out = self.sigmoid(self.fc(x_concat)).squeeze(-1)  # (batch_size,)
        return out


    # ---------------------- 新增：取句子特征的方法 ----------------------
    def extract_feature(self, x):
        # 和forward逻辑一致，只返回分类前的256维句子特征（x_concat）
        x_embed = self.embedding(x)
        x_embed = x_embed.permute(0, 2, 1)
        x1 = self.pool(torch.relu(self.conv1(x_embed))).squeeze(-1)
        x2 = self.pool(torch.relu(self.conv2(x_embed))).squeeze(-1)
        return torch.cat([x1, x2], dim=1)  # 返回256维句子特征


# --------------------------
# 4. 训练与评估函数
# --------------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 训练模式
    total_loss = 0.0
    # 加leave=False，避免进度条残留；desc改得更简洁
    for seq, label in tqdm(train_loader, desc="训练批次", leave=False):
        seq, label = seq.to(device), label.to(device)
        optimizer.zero_grad()  # 清空梯度
        pred = model(seq)  # 预测:model(seq)自动触发forward函数（核心计算），输出 32 个预测概率（shape [32]，0~1 之间，越接近 1 越可能是正面）
        loss = criterion(pred, label)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重:按梯度调整模型所有权重（卷积层、全连接层等）
        total_loss += loss.item() * seq.size(0)
    return total_loss / len(train_loader.dataset)  # 平均损失


def evaluate(model, test_loader, criterion, device):  # 新增criterion参数（损失函数）
    model.eval()
    y_true, y_pred = [], []
    total_test_loss = 0.0  # 新增：统计测试损失
    with torch.no_grad():
        for seq, label in test_loader:
            seq, label = seq.to(device), label.to(device)
            pred = model(seq)
            # 计算测试损失
            loss = criterion(pred, label)
            total_test_loss += loss.item() * seq.size(0)

            pred_np = pred.cpu().numpy()
            y_pred.extend((pred_np > 0.5).astype(int))
            y_true.extend(label.cpu().numpy())

    # 计算平均测试损失和整体准确率
    avg_test_loss = total_test_loss / len(test_loader.dataset)
    accuracy = (np.array(y_true) == np.array(y_pred)).mean()


    return avg_test_loss, accuracy  # 返回测试损失和准确率


# --------------------------
# 5. 主函数（流程入口）
# --------------------------
def main():
    # 配置参数（与word2vec.py保持一致）
    DATA_PATH = "dataset/dev.csv"  # 同Word2Vec使用的数据集
    W2V_MODEL_PATH = "word2vec_sentiment.model"  # 同目录下的Word2Vec模型
    EMBED_DIM = 100  # 必须与Word2Vec的vector_size一致
    BATCH_SIZE = 32
    EPOCHS = 7
    LEARNING_RATE = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备

    # 步骤1：加载数据
    print("加载并预处理数据...")
    corpus, labels = load_data(DATA_PATH)
    print(f"数据加载完成 | 总样本数: {len(corpus)} | 负面样本: {sum(labels == 0)} | 正面样本: {sum(labels == 1)}")

    # 步骤2：构建词汇表和序列
    print("构建词汇表...")
    tokenizer = Tokenizer(oov_token="<OOV>", filters="", lower=False)  # 与预处理逻辑匹配
    tokenizer.fit_on_texts(corpus)
    vocab_size = len(tokenizer.word_index) + 1  # +1预留padding索引

    # 自动计算最佳序列长度（95%分位数）
    seq_lengths = [len(tokens) for tokens in corpus]
    max_seq_len = int(np.percentile(seq_lengths, 95))
    print(f"词汇表大小: {vocab_size} | 序列最大长度: {max_seq_len}")

    # 步骤3：划分训练集和测试集
    print("划分训练集和测试集...")
    train_corpus, test_corpus, train_labels, test_labels = train_test_split(
        corpus, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 步骤4：创建数据集和数据加载器
    train_dataset = SentimentDataset(train_corpus, train_labels, tokenizer, max_seq_len)
    test_dataset = SentimentDataset(test_corpus, test_labels, tokenizer, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 步骤5：初始化模型
    print("初始化TextCNN模型...")
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        w2v_model_path=W2V_MODEL_PATH,
        tokenizer=tokenizer  # 新增这一行，传递tokenizer
        #tokenizer的核心是 “文字→数字” 的转换器，没有它，模型无法理解原始文本；有了它，文本才能变成模型能计算的数字序列，同时让嵌入层正确匹配预训练的词向量。
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 步骤6：训练模型
    print(f"开始训练（设备: {device}）...")
    # 保存每轮的测试损失和准确率，方便后续看趋势
    test_loss_history = []
    test_acc_history = []

    # ---------------------- 新增：早停初始化 ----------------------
    best_test_loss = float('inf')  # 记录最优测试损失（初始设为无穷大）
    patience = 3                 # 连续2轮测试损失上升就停止
    patience_counter = 0           # 计数连续上升的轮数
    best_model_path = "best_textcnn_model.pth"  # 最优模型保存路径

    # 初始化绘图用的列表
    epoch_list = []  # 轮数
    train_loss_list = []  # 每轮训练损失
    test_loss_list = []  # 每轮测试损失
    test_acc_list = []  # 每轮测试准确率

    for epoch in range(EPOCHS):
        # 训练并获取训练损失
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # 评估测试集，获取测试损失和准确率
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 记录本轮数据
        epoch_list.append(epoch + 1)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        # 保存历史数据
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        # ---------------------- 新增：早停判断 ----------------------
        # 打印每轮的核心指标（重点对比训练损失和测试损失）
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
        print(f"训练损失: {train_loss:.4f} | 测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f}")

        # 如果本轮测试损失比最优损失更低 → 更新最优，重置计数器
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # 保存当前最优模型（覆盖之前的）
            torch.save(model.state_dict(), best_model_path)
            print(f" 测试损失下降，保存最优模型到 {best_model_path}")
        # 否则 → 计数器+1，判断是否触发早停
        else:
            patience_counter += 1
            print(f"测试损失上升，连续上升轮数: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n测试损失连续{patience}轮上升，提前停止训练！")
                break  # 终止训练循环
        # -------------------------------------------------------------




    # 保存模型（可选）
    torch.save(model.state_dict(), "textcnn_model.pth")
    print("\n模型已保存为: textcnn_model.pth")

    # 步骤7：可视化调参
    # 设置画布大小和风格
    plt.figure(figsize=(10, 6))
    plt.style.use('default')

    # 绘制损失曲线（左轴）
    ax1 = plt.gca()  # 获取主坐标轴
    color1 = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color1)
    ax1.plot(epoch_list, train_loss_list, marker='o', color=color1, label='Train Loss', linewidth=2)
    ax1.plot(epoch_list, test_loss_list, marker='s', color=color1, linestyle='--', label='Test Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(alpha=0.3)

    # 绘制准确率曲线（右轴，双轴）
    ax2 = ax1.twinx()  # 创建次坐标轴
    color2 = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color2)
    ax2.plot(epoch_list, test_acc_list, marker='^', color=color2, label='Test Accuracy', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # 添加图例（合并两个轴的图例）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 添加标题和标注（重点标出过拟合拐点）
    plt.title('TextCNN: Train/Test Loss + Test Accuracy (10000 Samples)')


    # 保存+显示
    plt.savefig('textcnn_10000data_trend.png', dpi=150, bbox_inches='tight')
    plt.show()



    # 步骤8：可视化——句子向量TSNE降维散点图

    # ---------------------- 新增：情感特征TSNE可视化（测试集2000条） ----------------------
    print("\n生成情感特征可视化图...")
    # 1. 加载最优模型（用训练好的最优权重，保证特征质量）
    best_model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        w2v_model_path=W2V_MODEL_PATH,
        tokenizer=tokenizer
    ).to(device)
    best_model.load_state_dict(torch.load("best_textcnn_model.pth"))
    best_model.eval()  # 评估模式，不更新权重

    # 2. 提取测试集的句子特征+真实标签（批量提取，不占内存）
    test_features = []
    test_true_labels = []
    with torch.no_grad():  # 关闭梯度，提速
        for seq, label in tqdm(test_loader, desc="提取测试集特征"):
            seq = seq.to(device)
            feat = best_model.extract_feature(seq)  # 取256维特征
            test_features.extend(feat.cpu().numpy())  # 转numpy存起来
            test_true_labels.extend(label.numpy())  # 存真实标签

    # 3. TSNE降维（256维→2维，适配2000条数据，perplexity设30最优）
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    test_feat_2d = tsne.fit_transform(np.array(test_features))  # 降维后2000×2矩阵

    # 4. 画散点图（负面红、正面蓝，清晰区分）
    plt.figure(figsize=(10, 8))
    # 负面样本（label=0）
    neg_mask = np.array(test_true_labels) == 0
    plt.scatter(
        test_feat_2d[neg_mask, 0], test_feat_2d[neg_mask, 1],
        c='#FF6B6B', alpha=0.6, s=30, label='Negative (0)', edgecolors='none'
    )
    # 正面样本（label=1）
    pos_mask = np.array(test_true_labels) == 1
    plt.scatter(
        test_feat_2d[pos_mask, 0], test_feat_2d[pos_mask, 1],
        c='#4ECDC4', alpha=0.6, s=30, label='Positive (1)', edgecolors='none'
    )

    # 图注美化
    plt.xlabel('TSNE Dimension 1', fontsize=12)
    plt.ylabel('TSNE Dimension 2', fontsize=12)
    plt.title('TextCNN Sentiment Feature Visualization (10k Samples Test Set)', fontsize=14, pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('sentiment_feature_tsne.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("可视化图已保存为 sentiment_feature_tsne.png")



if __name__ == "__main__":
    main()
