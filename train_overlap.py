import torch
import torch.nn as nn
import pandas as pd
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
import itertools

class EmbeddingModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embeddings = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
    
    def forward(self, user_ids, item_ids, ratings):
        user_vecs = self.user_embeddings(user_ids)
        item_vecs = self.item_embeddings(item_ids)
        preds = (user_vecs * item_vecs).sum(dim=1)
        loss = nn.functional.mse_loss(preds, ratings)
        return preds, loss

#训练聚类模块
class ClusteringModule(nn.Module):
    def __init__(self, trained_user_embeddings1, trained_user_embeddings2, overlap_indices1, overlap_indices2, K, embedding_dim):
        super(ClusteringModule, self).__init__()
        self.K = K
        self.embedding_dim =  embedding_dim
        self.user_embeddings1 = nn.Parameter(trained_user_embeddings1)
        self.user_embeddings2 = nn.Parameter(trained_user_embeddings2)
        self.cluster_centers = nn.Parameter(torch.randn(K, embedding_dim))
        # 存储重叠用户在两个数据集中的索引
        self.overlap_indices1 = overlap_indices1
        self.overlap_indices2 = overlap_indices2

    def forward(self):
        # 计算P和Q
        # 由于sin距离在PyTorch中不是内置的，我们使用cosine相似度作为替代
        dist_matrix1 = torch.cdist(self.user_embeddings1, self.cluster_centers, p=2)
        dist_matrix2 = torch.cdist(self.user_embeddings2, self.cluster_centers, p=2)
        column_sums1 = dist_matrix1.sum(dim=0)
        column_sums2 = dist_matrix2.sum(dim=0)
        P_j = (column_sums1 - column_sums1.min()) / (column_sums1.max() - column_sums1.min() + 1e-5)
        Q_j = (column_sums2 - column_sums2.min()) / (column_sums2.max() - column_sums2.min() + 1e-5)      
        
        #print(P_j)
        #print(Q_j)
        # 计算KL散度
        L_IDA = torch.norm(P_j - Q_j, p=2)
        #KL_PQ = F.kl_div(P_j.log(), Q_j, reduction='batchmean')
        #KL_QP = F.kl_div(Q_j.log(), P_j, reduction='batchmean')
        
        #L_IDA = 0.5 * (KL_PQ + KL_QP)
        
        #print(f'L_IDA  ')


         # 计算所有中心之间的距离平方
        dist_matrix = torch.cdist(self.cluster_centers, self.cluster_centers, p=2) ** 2
        # 由于每个中心与自身的距离为0，需要将其排除在外
        mask = 1 - torch.eye(self.K, device= self.cluster_centers.device)
        dist_matrix = dist_matrix * mask
        # 计算中心之间的距离平方和的平均值
        term1 = dist_matrix.sum() / (self.K * (self.K - 1))

        # 计算用户嵌入和最近的类簇中心之间的距离平方
        # 合并两部分用户嵌入
        user_embeddings = torch.cat((self.user_embeddings1, self.user_embeddings2), dim=0)
    
        dist_to_centers = torch.cdist(user_embeddings, self.cluster_centers, p=2) ** 2
        # 取每个用户到其最近中心的距离平方
        min_dists, _ = dist_to_centers.min(dim=1)
        # 计算这些距离平方和的平均值
        term2 = min_dists.sum() / user_embeddings.size(0)
        # 损失是两部分的和
        L_cluster = term1 + term2 / self.K
        
        # 计算重叠用户嵌入一致性损失
        overlap_embeddings1 = self.user_embeddings1[self.overlap_indices1]
        overlap_embeddings2 = self.user_embeddings2[self.overlap_indices2]
        L_overlap = torch.norm(overlap_embeddings1 - overlap_embeddings2, p=2, dim=1).mean()
        
        return L_IDA, L_cluster,L_overlap

def train_clustering_module(module, optimizer, epochs=1):
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        L_IDA, L_cluster,L_overlap = module()
        
        # 总损失是两个损失的和
        total_loss = L_IDA + L_cluster+L_overlap
        
        total_loss.backward()
        optimizer.step()
        
        #print(f'Epoch {epoch+1}, Total Loss: {total_loss.item()}')

def generate_top_k(pred_ratings, k):
    """
    为每个用户生成top-k推荐列表。

    :param pred_ratings: 预测评分的张量，尺寸为[num_users, num_items]。
    :param k: 推荐列表中的项目数量。
    :return: 每个用户的top-k推荐列表，列表中的元素是项目的索引。
    """
    # 按评分降序排列，只获取前k个最高评分的索引
    top_k_preds = torch.topk(pred_ratings, k, dim=1)[1]
    return top_k_preds

def calculate_hr(top_k_preds, actual_item):
    """
    计算HR指标
    :param top_k_preds: top-k推荐列表，包含项目的索引
    :param actual_item: 用户实际感兴趣的项目的索引
    :return: HR值，1表示命中，0表示未命中
    """
    return 1 if actual_item in top_k_preds else 0


def calculate_ndcg(top_k_preds, actual_item):
    """
    计算NDCG指标
    :param top_k_preds: top-k推荐列表，包含项目的索引
    :param actual_item: 用户实际感兴趣的项目的索引
    :return: NDCG值
    """
    # 查找actual_item在top_k_preds中的位置
    matches = torch.eq(top_k_preds, actual_item)
    non_zero_indices = torch.nonzero(matches, as_tuple=True)[0]

    if len(non_zero_indices) > 0:
        index = non_zero_indices[0].item() + 1  # 加1是因为位置索引应从1开始计算
        return np.log(2) / np.log(index + 1)  # 使用index+1是因为NDCG的分母是以2为底的对数(index+1)
    else:
        return 0.0

# 保存用户嵌入和类簇嵌入
def save_embeddings(user_embeddings1, user_embeddings2, cluster_centers, save_dir="embeddings"):
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
    
    # 保存用户嵌入
    torch.save(user_embeddings1, os.path.join(save_dir, "user_embeddings1.pt"))
    torch.save(user_embeddings2, os.path.join(save_dir, "user_embeddings2.pt"))
    
    # 保存类簇嵌入
    torch.save(cluster_centers, os.path.join(save_dir, "cluster_centers.pt"))
    
    print("Embeddings saved successfully.")



# 主执行逻辑
if __name__ == "__main__":

    

    # 假设这是你的数据集路径
    train_path1 = r'E:\anaconda\workspace\ICLCDR\data\Amazon_5core\train\Music_train.csv'
    train_path2 = r'E:\anaconda\workspace\ICLCDR\data\Amazon_5core\train\Video_train.csv'
    test_path1 = r'E:\anaconda\workspace\ICLCDR\data\Amazon_5core\test\Music_test.csv'
    test_path2 = r'E:\anaconda\workspace\ICLCDR\data\Amazon_5core\test\Video_test.csv'
    # 读取数据集
    train_df1 = pd.read_csv(train_path1, header=None, names=['ItemID', 'UserID', 'Rating'])
    train_df2 = pd.read_csv(train_path2, header=None, names=['ItemID', 'UserID', 'Rating'])
    test_df1 = pd.read_csv(test_path1, header=None, names=['ItemID', 'UserID', 'Rating'])
    test_df2 = pd.read_csv(test_path2, header=None, names=['ItemID', 'UserID', 'Rating'])

    # 获取用户和项目的唯一ID数量
    num_users1 = len(set(train_df1['UserID']).union(set(train_df1['UserID'])))
    num_items1 = len(set(train_df1['ItemID']).union(set(train_df1['ItemID'])))
    num_users2 = len(set(train_df2['UserID']).union(set(train_df2['UserID'])))
    num_items2 = len(set(train_df2['ItemID']).union(set(train_df2['ItemID'])))

    # 1域初始化嵌入向量
    embedding_dim = 10
    user_embeddings1 = nn.Embedding(num_embeddings=num_users1, embedding_dim=embedding_dim)
    item_embeddings1 = nn.Embedding(num_embeddings=num_items1, embedding_dim=embedding_dim)
     # 2域初始化嵌入向量
    user_embeddings2 = nn.Embedding(num_embeddings=num_users2, embedding_dim=embedding_dim)
    item_embeddings2 = nn.Embedding(num_embeddings=num_items2, embedding_dim=embedding_dim)

    # 假设df是包含所有数据的DataFrame，包括训练集和可能的验证/测试集，以确保索引覆盖所有可能的ID
    df1 = pd.concat([train_df1, test_df1])
    df2 = pd.concat([train_df2, test_df2])
    # 创建用户ID到索引的映射
    user_ids1 = df1['UserID'].unique()
    user_to_index1 = {user_id: index for index, user_id in enumerate(user_ids1)}

    # 创建项目ID到索引的映射
    item_ids1 = df1['ItemID'].unique()
    item_to_index1 = {item_id: index for index, item_id in enumerate(item_ids1)}
    # 创建用户ID到索引的映射
    user_ids2 = df2['UserID'].unique()
    user_to_index2 = {user_id: index for index, user_id in enumerate(user_ids2)}

    # 创建项目ID到索引的映射
    item_ids2 = df2['ItemID'].unique()
    item_to_index2 = {item_id: index for index, item_id in enumerate(item_ids2)}

    # 准备第一个训练集的数据
    train_user_indices1 = torch.tensor([user_to_index1[uid] for uid in train_df1['UserID']], dtype=torch.long)
    train_item_indices1 = torch.tensor([item_to_index1[iid] for iid in train_df1['ItemID']], dtype=torch.long)
    train_ratings1 = torch.tensor(train_df1['Rating'].values, dtype=torch.float)

    # 准备第二个训练集的数据
    train_user_indices2 = torch.tensor([user_to_index2[uid] for uid in train_df2['UserID']], dtype=torch.long)
    train_item_indices2 = torch.tensor([item_to_index2[iid] for iid in train_df2['ItemID']], dtype=torch.long)
    train_ratings2 = torch.tensor(train_df2['Rating'].values, dtype=torch.float)
    
    # 定义参数空间
    param_space = {
        'lr': [0.001, 0.01, 0.1],
        'embedding_dim': [10, 20],
        'K': [5, 10]
    }

    # 准备记录最佳参数和分数
    best_params = None
    best_score = float('inf')  # 假设更低的损失值更好，可以根据实际情况调整

    # 生成所有参数组合
    all_params = list(itertools.product(*param_space.values()))

    for params in all_params:
        # 解包当前参数组合
        lr, embedding_dim, K = params
        
        # 在此处插入模型初始化和训练的代码
        # 你需要使用上面解包的参数来配置模型和训练过程

        
        # 训练模型并计算分数（例如，这里可以使用AUC值）
         # 实例化两个模型，一个用于每个训练集
        
        model1 = EmbeddingModel(num_users=num_users1, num_items=num_items1, embedding_dim=embedding_dim)
        model2 = EmbeddingModel(num_users=num_users2, num_items=num_items2, embedding_dim=embedding_dim)

        # 定义损失函数和优化器
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

        # 训练轮次
        epochs = 400

        # 训练第一个模型
        for epoch in range(epochs):
            optimizer1.zero_grad()
            preds1, loss1 = model1(train_user_indices1, train_item_indices1, train_ratings1)
            loss1.backward()
            optimizer1.step()
            #print(f"Epoch {epoch+1}, Domain 1 Loss: {loss1.item()}")

        # 训练第二个模型
        for epoch in range(epochs):
            optimizer2.zero_grad()
            preds2, loss2 = model2(train_user_indices2, train_item_indices2, train_ratings2)
            loss2.backward()
            optimizer2.step()
            #print(f"Epoch {epoch+1}, Domain 2 Loss: {loss2.item()}")

    
        # 假设model1和model2是两个训练完成的模型实例
        trained_user_embeddings1 = model1.user_embeddings.weight.data
        trained_user_embeddings2 = model2.user_embeddings.weight.data

        # trained_item_embeddings1和trained_item_embeddings2 包含每个域的item_embeddings
        trained_item_embeddings1 = model1.item_embeddings.weight.data
        trained_item_embeddings2 = model2.item_embeddings.weight.data
        
        #找重叠用户嵌入
        # 找到两个数据集中都出现过的用户ID
        overlap_user_ids = set(user_ids1).intersection(set(user_ids2))

        # 初始化重叠用户嵌入向量的字典
        overlap_user_embeddings1 = {}
        overlap_user_embeddings2 = {}

        # 初始化重叠用户索引列表
        overlap_indices1 = []
        overlap_indices2 = []
        # 对于每个重叠用户，提取其在两个数据集中的嵌入
        for user_id in overlap_user_ids:
            index1 = user_to_index1.get(user_id)  # 找到该用户在数据集1中的索引
            index2 = user_to_index2.get(user_id)  # 找到该用户在数据集2中的索引
            # 提取对应的嵌入向量
            if index1 is not None and index2 is not None and index1 < trained_user_embeddings1.size(0) and index2 < trained_user_embeddings2.size(0):
                embedding1 = trained_user_embeddings1[index1]
                embedding2 = trained_user_embeddings2[index2]
                 # 存储嵌入向量
                overlap_user_embeddings1[user_id] = embedding1
                overlap_user_embeddings2[user_id] = embedding2
                overlap_indices1.append(index1)
                overlap_indices2.append(index2)

            # 这里，overlap_user_embeddings1 和 overlap_user_embeddings2 分别包含了重叠用户在两个数据集中的嵌入表示
            else:
        # 这个user_id对应的索引超出了范围，所以跳过处理
                print(f"Skipping user_id {user_id} due to out of bounds index.")



        # 将重叠用户嵌入向量字典转换为DataFrame
        df_overlap_user_embeddings1 = pd.DataFrame.from_dict(overlap_user_embeddings1, orient='index')
        df_overlap_user_embeddings2 = pd.DataFrame.from_dict(overlap_user_embeddings2, orient='index')

        # 重置DataFrame的索引，使用户ID成为一个单独的列
        df_overlap_user_embeddings1.reset_index(inplace=True)
        df_overlap_user_embeddings2.reset_index(inplace=True)

        # 重命名列，以便第一列现在包含用户ID
        df_overlap_user_embeddings1.columns = ['UserID'] + [f'Feature_{i}' for i in range(df_overlap_user_embeddings1.shape[1] - 1)]
        df_overlap_user_embeddings2.columns = ['UserID'] + [f'Feature_{i}' for i in range(df_overlap_user_embeddings2.shape[1] - 1)]

        # 保存DataFrame为CSV文件
        df_overlap_user_embeddings1.to_csv("overlap_user_embeddings1.csv", index=False)
        df_overlap_user_embeddings2.to_csv("overlap_user_embeddings2.csv", index=False)

        print("Overlap user embeddings saved as CSV files.")

        # 实例化模块
        clustering_module = ClusteringModule(trained_user_embeddings1, trained_user_embeddings2, overlap_indices1, overlap_indices2,K, embedding_dim)

        # 定义优化器
        optimizer = torch.optim.Adam(clustering_module.parameters(), lr=0.001)

        # 训练
        train_clustering_module(clustering_module, optimizer, epochs=50)





        #评估指标RECALL和AUC
        
        # 获取训练完成后的用户嵌入表示
        final_user_embeddings1 = clustering_module.user_embeddings1.data
        final_user_embeddings2 = clustering_module.user_embeddings2.data

        clustering_embed = clustering_module.cluster_centers.data

        # 假设final_user_embeddings1和final_user_embeddings2是从PyTorch模型获取的张量
        # 并且你已经有了这些张量

        # 首先，将PyTorch张量转换为NumPy数组
        final_user_embeddings1_np = final_user_embeddings1.cpu().detach().numpy()
        final_user_embeddings2_np = final_user_embeddings2.cpu().detach().numpy()
        clustering_embed_np = clustering_embed.cpu().detach().numpy()

        # 使用Pandas将NumPy数组保存为CSV文件
        df_final_user_embeddings1 = pd.DataFrame(final_user_embeddings1_np)
        df_final_user_embeddings2 = pd.DataFrame(final_user_embeddings2_np)
        df_clustering_embed = pd.DataFrame(clustering_embed_np)

        # 指定要保存的文件名
        filename1 = "final_user_embeddings1.csv"
        filename2 = "final_user_embeddings2.csv"
        filename3 = "final_user_embeddings3.csv"
        # 保存为CSV文件，不保存行索引
        df_final_user_embeddings1.to_csv(filename1, index=False)
        df_final_user_embeddings2.to_csv(filename2, index=False)
        df_clustering_embed.to_csv(filename3, index=False)

        print(f"Final user embeddings saved as CSV files: {filename1}, {filename2},{filename3}")

        
        # 将测试集用户ID和项目ID转换为索引
        test_user_indices = [user_to_index1.get(uid, -1) for uid in test_df1['UserID']]
        test_item_indices = [item_to_index1.get(iid, -1) for iid in test_df1['ItemID']]

        # 过滤掉测试集中未知的用户或项目
        known_indices = [(u, i) for u, i in zip(test_user_indices, test_item_indices) if u != -1 and i != -1]

        # 分别提取已知用户和项目的索引
        known_user_indices = torch.tensor([u for u, i in known_indices], dtype=torch.long)
        known_item_indices = torch.tensor([i for u, i in known_indices], dtype=torch.long)

        # 使用final_user_embeddings1和trained_item_embeddings1生成预测
        user_embeddings = final_user_embeddings1[known_user_indices]
        item_embeddings = trained_item_embeddings1[known_item_indices]
        pred_ratings = torch.sum(user_embeddings * item_embeddings, dim=1) 
        
        # 示例：假设k为10
       
        # 生成每个用户的top-k推荐列表
        top_k_preds = torch.topk(pred_ratings, k=10,dim = 0)[1]
        # 初始化HR和NDCG得分列表
        hr_scores = []
        ndcg_scores = []

        for i in range(known_user_indices.size(0)):
            actual_item = known_item_indices[i].item()
            
            # 获取当前用户的top-k推荐项目索引
            top_k_preds = torch.topk(pred_ratings, k=10,dim = 0)[1]
            
            # 计算HR和NDCG
            hr_score = calculate_hr(top_k_preds, actual_item)
            ndcg_score = calculate_ndcg(top_k_preds, actual_item)
            
            hr_scores.append(hr_score)
            ndcg_scores.append(ndcg_score)

        # 计算平均HR和NDCG
        mean_hr = np.mean(hr_scores)
        mean_ndcg = np.mean(ndcg_scores)

        print(f"Mean HR@{10}: {mean_hr:.4f}")
        print(f"Mean NDCG@{20}: {mean_ndcg:.4f}")



        # 实际评分
        actual_ratings = torch.tensor([test_df1.iloc[i]['Rating'] for _, i in known_indices])

        # 计算真实的标签：评分 > 2.0 视为正样本
        true_labels = (actual_ratings > 2.0).numpy().astype(int)

        # 预测的标签基于评分阈值：这里使用2.0作为示例，实际应根据预测评分调整
        pred_labels = (pred_ratings > 2.0).numpy().astype(int)

        # 计算Recall和AUC值
        recall = recall_score(true_labels, pred_labels)
        auc = roc_auc_score(true_labels, pred_ratings.numpy())

        #print(f'Recall: {recall:.4f}')
        #print(f'AUC: {auc:.4f}')

        current_score = auc  # 伪代码，实际中应替换为计算分数的代码
        
        # 更新最佳参数和分数
        if current_score < best_score:
            best_score = current_score
            best_params = {
                'lr': lr,
                'embedding_dim': embedding_dim,
                'K': K
            }

    print("最佳参数组合:", best_params)
    print("最佳分数:", best_score)

    # 调用函数来保存嵌入
    save_embeddings(final_user_embeddings1, final_user_embeddings2, clustering_module.cluster_centers.data)
