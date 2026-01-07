# ====================== 1. 导入依赖库（工具准备） ======================
# networkx：用于构建图结构（节点+边）
import networkx as nx
import pandas as pd
from node2vec import Node2Vec
# gensim：用于保存/加载节点向量模型，处理Word2Vec格式数据
from gensim.models import KeyedVectors

# ====================== 2. 读取并预处理数据（数据准备） ======================

df = pd.read_excel('posting-test.xlsx')
# 数据预处理：解决公司名称缺失问题
df['company_name'] = df['company_name'].fillna('Unknown_Company').astype(str)
# 替换空格、斜杠、逗号等分隔符，避免保存时拆分节点名
df['company_name'] = df['company_name'].str.replace(r'[\s/,:()]+', '_', regex=True)
df['job_id'] = df['job_id'].astype(str).str.replace(r'[\s/,:()]+', '_', regex=True)

print(f"数据加载完成：共{len(df)}条职位记录")

# ====================== 3. 构建图结构（核心逻辑：实体+关系建模） ======================
# 初始化无向图：Graph()表示无向边（职位→公司 和 公司→职位是同一关系）
G = nx.Graph()

# 遍历每条职位记录，构建节点和边
for _, row in df.iterrows():
    # 3.1 定义节点：区分“职位节点”和“公司节点”
    job_node = row['job_id']  # 职位节点：用job_id唯一标识
    company_node = row['company_name']  # 公司节点：用company_name标识

    # 3.2 添加职位节点：附带type属性（标记节点类型，方便后续区分）
    G.add_node(
        job_node,
        type='job'  # 属性：标记节点为“职位”类型
    )

    # 3.3 添加公司节点：同样附带type属性（标记为“公司”类型）
    G.add_node(
        company_node,
        type='company'  # 属性：标记节点为“公司”类型
    )

    # 3.4 添加边：连接职位和公司，表示“该职位属于此公司”的关系
    # 边的作用：将孤立的节点关联，形成“职位-公司”的关系网
    G.add_edge(job_node, company_node)

# 验证图结构：输出关键统计信息，确保构建正确
print(f"\n图构建完成：")
print(f"- 总节点数：{G.number_of_nodes()}（职位节点+公司节点）")
print(f"- 总边数：{G.number_of_edges()}（每条边对应一个职位-公司关联）")
# 统计节点类型分布：检查职位/公司节点数量是否合理
node_types = pd.Series([G.nodes[n]['type'] for n in G.nodes]).value_counts()
print(f"- 节点类型分布：{node_types.to_dict()}")

# ======================= 4. 训练并保存模型 =======================
# 初始化Node2Vec模型
node2vec = Node2Vec(
    G,
    dimensions=64,      # 向量维度
    walk_length=30,     # 随机游走长度
    num_walks=200,      # 每个节点的游走次数
    workers=4           # 并行线程数
)

# 训练模型
model = node2vec.fit(
    window=10,
    min_count=1,
    batch_words=4,
    epochs=10           # 训练轮数
)

# 保存模型
model.save("node2vec_linkedIn.model")
print("\nNode2Vec模型已保存")



# ======================= 5. T-SNE 可视化 =======================
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 5.1 取出向量与对应节点名
node_ids = model.wv.index_to_key
vectors = model.wv.vectors

# 5.2 降维到 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
vectors_2d = tsne.fit_transform(vectors)

# 5.3 构造颜色标签
node_type = [G.nodes[n]['type'] for n in node_ids]   # 'job' or 'company'
color_map = {'job': 'tab:blue', 'company': 'tab:orange'}
colors = [color_map[t] for t in node_type]

# 5.4 画图
plt.figure(figsize=(8, 6))
sns.scatterplot(x=vectors_2d[:, 0],
                y=vectors_2d[:, 1],
                hue=node_type,
                palette=color_map,
                s=15,
                alpha=0.7)
plt.title('Node2Vec + T-SNE: Job vs Company')
plt.legend(title='Node Type')
plt.tight_layout()
plt.savefig("tsne_job_company.png", dpi=300)
plt.show()

# ======================= 6. 相似度计算示例 =======================
def most_similar_node(seed_name, topn=5):
    """返回与 seed_name 最相似的同类型节点"""
    if seed_name not in model.wv:
        print(f"{seed_name} 不在词汇表中")
        return
    seed_type = G.nodes[seed_name]['type']
    sims = model.wv.most_similar(seed_name, topn=topn*3)  # 多拿一点再过滤
    filtered = [(n, s) for n, s in sims if G.nodes[n]['type'] == seed_type][:topn]
    print(f"\n与 {seed_name}（{seed_type}）最相似的 {topn} 个同类节点：")
    for n, s in filtered:
        print(f"  {n:<40} 相似度={s:.3f}")

# 6.1 挑一个职位做演示
job_example = [n for n in node_ids if G.nodes[n]['type'] == 'job'][0]
most_similar_node(job_example)

# 6.2 挑一个公司做演示
company_example = [n for n in node_ids if G.nodes[n]['type'] == 'company'][0]
most_similar_node(company_example)