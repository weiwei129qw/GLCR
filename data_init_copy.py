import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, recall_score
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RatingPredictionModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(RatingPredictionModel, self).__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

    def forward(self, user_indices, item_indices):
        user_vecs = self.user_embedding(user_indices)
        item_vecs = self.item_embedding(item_indices)
        ratings_pred = (user_vecs * item_vecs).sum(dim=1)
        return ratings_pred

def prepare_dataset_and_initialize_embeddings(train_file_path, embedding_dim, device):
    train_data = pd.read_csv(train_file_path, header=None, names=['ItemID', 'UserID', 'Rating'])
    user_ids = train_data['UserID'].unique()
    item_ids = train_data['ItemID'].unique()
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    ratings_index = {(item_index[row['ItemID']], user_index[row['UserID']]): row['Rating']
                     for index, row in train_data.iterrows()}
    num_users = len(user_ids)
    num_items = len(item_ids)
    return train_data, num_users, num_items, ratings_index, user_index, item_index

def train_and_save_embeddings(train_file_path, embedding_dim, device, epochs, save_dir):
    train_data, num_users, num_items, ratings_index, user_index, item_index = prepare_dataset_and_initialize_embeddings(
        train_file_path, embedding_dim, device
    )
    model = RatingPredictionModel(num_users, num_items, embedding_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        all_user_indices = torch.tensor([ui for (_, ui) in ratings_index.keys()], dtype=torch.long).to(device)
        all_item_indices = torch.tensor([ii for (ii, _) in ratings_index.keys()], dtype=torch.long).to(device)
        all_ratings = torch.tensor(list(ratings_index.values()), dtype=torch.float).to(device)
        
        optimizer.zero_grad()
        ratings_pred = model(all_user_indices, all_item_indices)
        loss = loss_fn(ratings_pred, all_ratings)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training loss: {loss.item()}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.user_embedding.state_dict(), os.path.join(save_dir, 'user_embedding.pt'))
    torch.save(model.item_embedding.state_dict(), os.path.join(save_dir, 'item_embedding.pt'))
    
    # 保存用户和项目的索引映射
    with open(os.path.join(save_dir, 'user_index.pkl'), 'wb') as f:
        pickle.dump(user_index, f)
    with open(os.path.join(save_dir, 'item_index.pkl'), 'wb') as f:
        pickle.dump(item_index, f)

    # 保存索引到用户ID和项目ID的逆映射
    index_user = {index: user_id for user_id, index in user_index.items()}
    index_item = {index: item_id for item_id, index in item_index.items()}
    with open(os.path.join(save_dir, 'index_user.pkl'), 'wb') as f:
        pickle.dump(index_user, f)
    with open(os.path.join(save_dir, 'index_item.pkl'), 'wb') as f:
        pickle.dump(index_item, f)

    print("Training completed. Embeddings and indices have been saved.")


#聚类函数
class ExtendedClusterModule(nn.Module):
    def __init__(self, user_embedding_weights1, user_embedding_weights2, K, embedding_dim):
        super(ExtendedClusterModule, self).__init__()
        self.K = K
        self.embedding_dim = embedding_dim
        # 将用户嵌入初始化为可学习的参数
        self.user_embeddings1 = nn.Parameter(user_embedding_weights1)
        self.user_embeddings2 = nn.Parameter(user_embedding_weights2)
        # 初始化K个类簇中心为可学习的参数
        self.cluster_centers = nn.Parameter(torch.randn(K, embedding_dim))

    def forward(self):
        dist1 = torch.cdist(self.user_embeddings1, self.cluster_centers, p=2)  # 56158 * k
        dist2 = torch.cdist(self.user_embeddings2, self.cluster_centers, p=2)  # 3815 * k
        
        column_sum_dist1 = torch.sum(dist1, dim=0)
        column_sum_dist2 = torch.sum(dist2, dim=0)
        column_sum_dist1_min = column_sum_dist1.min()
        column_sum_dist1_max = column_sum_dist1.max()
        norm_dist1 = (column_sum_dist1 - column_sum_dist1_min) / (column_sum_dist1_max - column_sum_dist1_min)
        column_sum_dist2_min = column_sum_dist2.min()
        column_sum_dist2_max = column_sum_dist2.max()
        norm_dist2 = (column_sum_dist2 - column_sum_dist2_min) / (column_sum_dist2_max - column_sum_dist2_min)

        #norm_dist1 = F.softmax(-dist1, dim=1)
        #norm_dist2 = F.softmax(-dist2, dim=1)
        print(norm_dist1)
        print(norm_dist2)

        return norm_dist1, norm_dist2

    def compute_losses(self, norm_dist1, norm_dist2):
        # 计算双向KL散度损失 L_IDA
        kl_div1 = F.kl_div(norm_dist1.log(), norm_dist2, reduction='batchmean')
        kl_div2 = F.kl_div(norm_dist2.log(), norm_dist1, reduction='batchmean')
        L_IDA = (kl_div1 + kl_div2) / 2

        # 这里直接使用L_IDA作为示例损失，根据需求添加其他损失
        return L_IDA

def train_clustering(user_embedding_weights1, user_embedding_weights2, K, embedding_dim, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir_arts = '/root/ICLCDR/Data/Amazon_5core/embed_dir/Arts_train'
    save_dir_luxury = '/root/ICLCDR/Data/Amazon_5core/embed_dir/Luxury_train'
    cluster_module = ExtendedClusterModule(user_embedding_weights1.to(device), user_embedding_weights2.to(device), K, embedding_dim).to(device)
    optimizer = optim.Adam(cluster_module.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        norm_dist1, norm_dist2 = cluster_module()
        loss = cluster_module.compute_losses(norm_dist1, norm_dist2)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Total Loss: {loss.item()}")

    # 返回更新后的用户嵌入权重
    torch.save(cluster_module.user_embeddings1, save_dir_arts + '.pt')
    torch.save(cluster_module.user_embeddings2, save_dir_luxury + '.pt')

    print("Updated embeddings have been saved.")


class RatingPredictionModel1(nn.Module):
    def __init__(self, user_embedding, item_embedding):
        super(RatingPredictionModel1, self).__init__()
        # 假设 user_embedding_weights 是一个预先计算好的嵌入向量张量
        self.user_embedding = nn.Embedding.from_pretrained(user_embedding, freeze=False)
        self.item_embedding = nn.Embedding.from_pretrained(item_embedding, freeze=False)

    def forward(self, user_indices, item_indices):
        # 根据索引获取用户和项目的嵌入向量
        user_vecs = self.user_embedding(user_indices)
        item_vecs = self.item_embedding(item_indices)
        
        # 计算嵌入向量的点积来预测评分
        ratings_pred = (user_vecs * item_vecs).sum(dim=1)
        return ratings_pred


# 主执行逻辑
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 10
    epochs1 = 300
    K = 5
    epochs2 = 1
    train_file_path1 = '/root/ICLCDR/Data/Amazon_5core/train/Arts_train.csv'  # Update this path
    train_file_path2 = '/root/ICLCDR/Data/Amazon_5core/train/Luxury_train.csv'  # Update this path
    save_dir1 = '/root/ICLCDR/Data/Amazon_5core/embed_dir/Arts_train'  # Update this path if needed
    save_dir2 = '/root/ICLCDR/Data/Amazon_5core/embed_dir/Luxury_train'  # Update this path if needed
    
    #首次MF得到节点嵌入表示
    #train_and_save_embeddings(train_file_path1, embedding_dim, device, epochs1, save_dir1)
    #train_and_save_embeddings(train_file_path2, embedding_dim, device, epochs1, save_dir2)
    
    #加载首次MF得到节点嵌入表示路径
    user_embedding_path1 = os.path.join(save_dir1, 'user_embedding.pt')
    user_embedding_path2 = os.path.join(save_dir2, 'user_embedding.pt')
 
    # 假设user_embedding_weights1和user_embedding_weights2是从状态字典中直接加载的嵌入层权重
    user_embedding_weights1 = torch.load(user_embedding_path1)['weight']    #[56158, 10]
    user_embedding_weights2 = torch.load(user_embedding_path2)['weight']    #[3815, 10]
    #print(user_embedding_weights2.shape)

    
    # 传入聚类函数更新节点嵌入，并将结果保存到.pt文件中
    #train_clustering(user_embedding_weights1, user_embedding_weights2, K, embedding_dim, epochs2)
    
    



    # 加载用户嵌入和项目嵌入
    user_embedding_path = '/root/ICLCDR/Data/Amazon_5core/embed_dir/Arts_train.pt'
    item_embedding_path = '/root/ICLCDR/Data/Amazon_5core/embed_dir/Arts_train/item_embedding.pt'
    user_embedding = torch.load(user_embedding_path)
    item_embedding_dict = torch.load(item_embedding_path)
    
    # 假设状态字典中的嵌入层权重键为 'weight'，根据你的实际键名调整
    item_embedding = item_embedding_dict['weight']
    print(item_embedding.shape)
    
    test_file_path1 = '/root/ICLCDR/Data/Amazon_5core/test/Arts_test.csv'
    test_data = pd.read_csv(test_file_path1, header=None, names=['ItemID', 'UserID', 'Rating'])
    

    
    model = RatingPredictionModel1(user_embedding, item_embedding)
    model = model.to(device)


    # 将用户ID和项目ID映射到索引
    # 加载用户ID到索引和项目ID到索引的映射
    user_index_path = '/root/ICLCDR/Data/Amazon_5core/embed_dir/Arts_train/user_index.pkl'
    item_index_path = '/root/ICLCDR/Data/Amazon_5core/embed_dir/Arts_train/item_index.pkl'
    with open(user_index_path, 'rb') as f:
        test_user_indices = pickle.load(f)
    with open(item_index_path, 'rb') as f:
        test_item_indices = pickle.load(f)

    
    
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