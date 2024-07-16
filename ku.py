import pandas as pd

# 读取训练集和测试集数据
train_path1 = r'E:\anaconda\workspace\ICLCDR\data\Amazon_5core\train\Video_train.csv'
test_path1 = r'E:\anaconda\workspace\ICLCDR\data\Amazon_5core\test\Video_test.csv'
train_path2 = r'E:\anaconda\workspace\ICLCDR\data\Amazon_5core\train\Music_train.csv'
test_path2 = r'E:\anaconda\workspace\ICLCDR\data\Amazon_5core\test\Music_test.csv'

train_df1 = pd.read_csv(train_path1, header=None, names=['ItemID', 'UserID', 'Rating'])
test_df1 = pd.read_csv(test_path1, header=None, names=['ItemID', 'UserID', 'Rating'])
train_df2 = pd.read_csv(train_path2, header=None, names=['ItemID', 'UserID', 'Rating'])
test_df2 = pd.read_csv(test_path2, header=None, names=['ItemID', 'UserID', 'Rating'])

# 合并训练集和测试集
merged_df1 = pd.concat([train_df1, test_df1])
merged_df2 = pd.concat([train_df2, test_df2])
# 找到两个数据集中都出现过的用户ID
overlap_user_ids = set(merged_df1['UserID']).intersection(set(merged_df2['UserID']))


A_user_count = merged_df1['UserID'].nunique()
B_user_count = merged_df2['UserID'].nunique()
print("A数据集中的用户数:", A_user_count)
print("B数据集中的用户数:", B_user_count)

# 计算重叠用户数量
num_overlap_users = len(overlap_user_ids)
print("Number of overlapping users:", num_overlap_users)
# 计算Arts数据集的密度
A_user_count = merged_df1['UserID'].nunique()
A_item_count = merged_df1['ItemID'].nunique()
A_ratings_count = len(merged_df1)
A_density = A_ratings_count / (A_user_count * A_item_count)

print("A数据集密度:", A_density)

# 计算Luxury数据集的密度
B_user_count = merged_df2['UserID'].nunique()
B_item_count = merged_df2['ItemID'].nunique()
B_ratings_count = len(merged_df2)
B_density = B_ratings_count / (B_user_count * B_item_count)

print("B数据集密度:", B_density)
