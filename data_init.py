import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, recall_score

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# 使用.to(device)将tensor或模型移动到GPU
tensor = torch.rand(10).to(device)

# 读取文件路径
file_path = '/root/ICLCDR/Data/Amazon_5core/train/Arts_train.csv'
file_path1 = '/root/ICLCDR/Data/Amazon_5core/test/Arts_test.csv'
test_data = pd.read_csv(file_path1, header=None, names=['ItemID', 'UserID', 'Rating'])

# 重新读取文件，正确设置列名
data = pd.read_csv(file_path, header=None, names=['ItemID', 'UserID', 'Rating'])

# 为用户和Item构建索引
user_ids = data['UserID'].unique()
item_ids = data['ItemID'].unique()

# 创建用户和Item的索引字典
user_index = {user_id: index for index, user_id in enumerate(user_ids)}
item_index = {item_id: index for index, item_id in enumerate(item_ids)}

# 假设 data 是你的DataFrame，且 user_index 和 item_index 字典已经创建

# 初始化一个空字典来存储评分数据
ratings_index = {}

# 遍历DataFrame中的每一行
for index, row in data.iterrows():
    # 获取当前行的项目ID和用户ID
    item_id = row['ItemID']
    user_id = row['UserID']
    
    # 使用之前创建的索引字典找到对应的索引值
    item_idx = item_index[item_id]
    user_idx = user_index[user_id]
    
    # 创建键为(item_idx, user_idx)的元组，值为评分的字典条目
    ratings_index[(item_idx, user_idx)] = row['Rating']

# 现在 ratings_dict 包含了所有的评分数据


# 获取用户和项目的数量
num_users = len(user_ids)
num_items = len(item_ids)
# 嵌入向量的维度
embedding_dim = 10

# 嵌入层的初始化
user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

# 将嵌入层移到之前检测到的设备上（GPU或CPU）
user_embedding = user_embedding.to(device)
item_embedding = item_embedding.to(device)




class RatingPredictionModel(nn.Module):
    def __init__(self, user_embedding, item_embedding):
        super(RatingPredictionModel, self).__init__()
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding

    def forward(self, user_indices, item_indices):
        # 根据索引获取用户和项目的嵌入向量
        user_vecs = self.user_embedding(user_indices)
        item_vecs = self.item_embedding(item_indices)
        
        # 计算嵌入向量的点积来预测评分
        ratings_pred = (user_vecs * item_vecs).sum(dim=1)
        return ratings_pred

# 初始化模型
model = RatingPredictionModel(user_embedding, item_embedding)
model = model.to(device)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



# 将用户索引和项目索引转换为Tensor，然后移到适当的设备
all_user_indices = torch.tensor([user_index for (_, user_index) in ratings_index.keys()], dtype=torch.long).to(device)
all_item_indices = torch.tensor([item_index for (item_index, _) in ratings_index.keys()], dtype=torch.long).to(device)
all_ratings = torch.tensor(list(ratings_index.values()), dtype=torch.float).to(device)

epochs = 3  # 定义训练的总轮数

for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    
    # 前向传播
    all_ratings_pred = model(all_user_indices, all_item_indices)

    # 计算损失
    loss = loss_fn(all_ratings_pred, all_ratings)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印每轮的损失
    print(f"Epoch {epoch+1}/{epochs}, Training loss: {loss.item()}")


# 将用户ID和项目ID映射到索引
test_user_indices = torch.tensor([user_index.get(user_id, -1) for user_id in test_data['UserID']], dtype=torch.long).to(device)
test_item_indices = torch.tensor([item_index.get(item_id, -1) for item_id in test_data['ItemID']], dtype=torch.long).to(device)

# 使用模型进行评分预测
model.eval()  # 设置为评估模式
with torch.no_grad():  # 不计算梯度
    test_ratings_pred = model(test_user_indices, test_item_indices).cpu().numpy()

# 真实评分
test_ratings_true = test_data['Rating'].values

# 定义正样本的阈值
threshold = 2.0
predictions = (test_ratings_pred > threshold).astype(int)
true_labels = (test_ratings_true > threshold).astype(int)

# 计算AUC
auc = roc_auc_score(true_labels, test_ratings_pred)

# 计算召回率
recall = recall_score(true_labels, predictions)

print(f"AUC: {auc}")
print(f"Recall: {recall}")

