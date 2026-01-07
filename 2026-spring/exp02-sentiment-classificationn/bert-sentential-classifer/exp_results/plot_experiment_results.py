import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取实验结果文件
df = pd.read_csv(r'C:\Users\lh008\Desktop\exp02-sentiment-classificationn\bert-sentential-classifer\exp_results\bert_experiment_results.csv')


# ---------------- 2. 全局风格 ----------------
sns.set(style="whitegrid")
palette = sns.color_palette("tab10", 3)


# =========================================================
# 图 2：双曲线图（仅 test_loss + val_acc，双 y 轴）
# =========================================================
fig, ax1 = plt.subplots(figsize=(7, 4))

# 左轴：test_loss
sns.lineplot(data=df, x='epochs', y='test_loss',
             color=palette[1], label='test_loss', ax=ax1)
ax1.set_ylabel('Loss', color=palette[1])
ax1.tick_params(axis='y', labelcolor=palette[1])

# 右轴：val_acc
ax2 = ax1.twinx()
sns.lineplot(data=df, x='epochs', y='val_acc',
             color=palette[2], label='val_acc', ax=ax2)
ax2.set_ylabel('Accuracy', color=palette[2])
ax2.tick_params(axis='y', labelcolor=palette[2])

# 合并图例
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc='center right')

plt.title('Loss & Accuracy Trends')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

# =========================================================
# 图 3：箱线图（test_loss 与 val_acc）
# =========================================================
plt.figure(figsize=(5, 4))
box_df = pd.melt(df[['test_loss', 'val_acc']],
                 var_name='metric', value_name='value')
sns.boxplot(data=box_df,
            x='metric',
            y='value',
            hue='metric',
            palette=[palette[1], palette[2]],
            legend=False)
plt.title('Boxplot: Test Loss vs. Val Accuracy')
plt.ylabel('Value')
plt.tight_layout()
plt.show()

# =========================================================
# 图 4：随 train_size 变化的双曲线图（test_loss + val_acc）
# =========================================================
fig4, ax4_left = plt.subplots(figsize=(7, 4))

# 左轴：test_loss
sns.lineplot(data=df, x='train_size', y='test_loss',
             color=palette[1], marker='o', label='test_loss', ax=ax4_left)
ax4_left.set_ylabel('Test Loss', color=palette[1])
ax4_left.tick_params(axis='y', labelcolor=palette[1])

# 右轴：val_acc
ax4_right = ax4_left.twinx()
sns.lineplot(data=df, x='train_size', y='val_acc',
             color=palette[2], marker='s', label='val_acc', ax=ax4_right)
ax4_right.set_ylabel('Val Accuracy', color=palette[2])
ax4_right.tick_params(axis='y', labelcolor=palette[2])

# 合并图例
l4_1, lb4_1 = ax4_left.get_legend_handles_labels()
l4_2, lb4_2 = ax4_right.get_legend_handles_labels()
ax4_left.legend(l4_1 + l4_2, lb4_1 + lb4_2, loc='center right')

plt.title('Test Loss & Val Accuracy vs. Training Set Size')
plt.xlabel('Training Set Size (train_size)')
plt.tight_layout()
plt.show()