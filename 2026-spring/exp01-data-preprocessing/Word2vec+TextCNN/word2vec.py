import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re
from tqdm import tqdm  # 导入 tqdm 库
# 新增聚类和可视化相关库
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 下载必要的nltk数据
nltk.download('punkt')


def preprocess_text(text):
    """文本预处理函数"""
    # 确保输入为字符串类型
    if not isinstance(text, str):
        text = str(text)

    # 转换为小写
    text = text.lower()
    # 移除特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    return tokens


def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    # 显式指定列名
    columns = ['label', 'title', 'review']
    # 读取CSV文件，并指定列名
    df = pd.read_csv(file_path, names=columns, header=None)

    # 打印列名，检查列名是否正确
    print(df.columns)

    # 处理缺失值
    df = df.fillna('')  # 用空字符串填充缺失值

    # 合并标题和评论
    df['text'] = df['title'] + " " + df['review']

    # 预处理所有文本
    corpus = []
    for text in df['text']:
        tokens = preprocess_text(text)
        corpus.append(tokens)

    return corpus, df  # 返回处理后的文本和整个数据框


def train_word2vec(corpus):
    """训练Word2Vec模型，并显示进度条"""
    # 用 tqdm 包裹 corpus 数据，显示进度条
    model = Word2Vec(sentences=tqdm(corpus, desc="Training Word2Vec", total=len(corpus), unit="sentence"),
                     vector_size=100,  # 词向量维度
                     window=5,  # 上下文窗口大小
                     min_count=1,  # 词频阈值
                     workers=4)  # 训练的线程数
    return model


def get_document_vector(text, model):
    """获取文档的词向量表示（取平均）"""
    tokens = preprocess_text(text)
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


def visualize_word_clusters(model, top_n=100, n_clusters=5):
    """可视化词向量聚类结果"""
    # 获取高频词及其向量
    words = model.wv.index_to_key[:top_n]  # 取前top_n个高频词
    word_vectors = np.array([model.wv[word] for word in words])

    # 降维（先用PCA降维到50维，再用TSNE降维到2维）
    pca = PCA(n_components=50)
    word_vectors_pca = pca.fit_transform(word_vectors)
    tsne = TSNE(n_components=2, random_state=42, perplexity=49)
    word_vectors_tsne = tsne.fit_transform(word_vectors_pca)

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(word_vectors_tsne)

    # 绘制聚类散点图
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(word_vectors_tsne[:, 0], word_vectors_tsne[:, 1],
                          c=clusters, cmap='viridis', alpha=0.7)

    # 标注部分重要词汇
    for i, word in enumerate(words):
        if i % 5 == 0:  # 每5个词标注一个，避免过于拥挤
            plt.annotate(word, (word_vectors_tsne[i, 0], word_vectors_tsne[i, 1]),
                         fontsize=9, alpha=0.8)

    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Word2Vec Word Clustering (Top {top_n} Words, {n_clusters} Clusters)')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.tight_layout()
    plt.savefig('word_clusters.png', dpi=300)
    plt.show()




def main():
    # 加载数据并预处理
    corpus, df = load_and_preprocess_data('dataset/dev.csv')

    # 训练Word2Vec模型
    model = train_word2vec(corpus)

    # 获取文档向量
    doc_vectors = []
    for text in tqdm(df['text'], desc="Processing documents", unit="document"):
        doc_vector = get_document_vector(text, model)
        doc_vectors.append(doc_vector)

    # 转换为numpy数组
    X = np.array(doc_vectors)
    y = df['label'].values  # 使用处理后的标签

    print("文档向量形状:", X.shape)
    print("标签形状:", y.shape)

    # 保存模型（可选）
    model.save("word2vec_sentiment.model")

    # 示例：查看某些词的相似词
    word = "great"
    if word in model.wv:
        similar_words = model.wv.most_similar(word)
        print(f"\n与'{word}'最相似的词:")
        for word, score in similar_words:
            print(f"{word}: {score:.4f}")

    # 词向量聚类可视化
    print("\nGenerating word cluster visualization...")
    visualize_word_clusters(model, top_n=150, n_clusters=5)




if __name__ == "__main__":
    main()