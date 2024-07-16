import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import re
feature_cols = ['Feature_' + str(i) for i in range(20)]
# 加载数据集1和2（已经是纯数值，无需转换）
df1 = pd.read_csv('E:/anaconda/workspace/final_user_embeddings1.csv', header=None, names=feature_cols)
df2 = pd.read_csv('E:/anaconda/workspace/final_user_embeddings2.csv', header=None, names=feature_cols)

# 为df1和df2添加标签列
df1['label'] = 'Dataset1'
df2['label'] = 'Dataset2'

# 加载数据集3和4（需要转换）
def load_and_convert_tensor_format(path):
    df = pd.read_csv(path)
    for column in df.columns[1:]:  # 跳过UserID列，从Feature_0开始转换
        df[column] = df[column].apply(lambda x: float(re.search(r'tensor\((.*?)\)', x).group(1)) if pd.notnull(x) else np.nan)
    return df

df3 = load_and_convert_tensor_format('E:/anaconda/workspace/overlap_user_embeddings1.csv')
df4 = load_and_convert_tensor_format('E:/anaconda/workspace/overlap_user_embeddings2.csv')

# 为df3和df4添加标签列
df3['label'] = 'Dataset3'
df4['label'] = 'Dataset4'

# 合并所有DataFrame
df_all = pd.concat([df1, df2, df3.drop('UserID', axis=1), df4.drop('UserID', axis=1)])

# 分离特征和标签
X = df_all.drop(columns=['label']).values
y = df_all['label'].values

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 可视化
colors = {'Dataset1': 'r', 'Dataset2': 'g', 'Dataset3': 'b', 'Dataset4': 'c'}
plt.figure(figsize=(8, 6))
for label, color in colors.items():
    idx = y == label
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=color, label=label, alpha=0.7)
plt.legend()
plt.title('Node Embeddings Visualization')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.show()
