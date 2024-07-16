import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个新的3D项目图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 示例数据
x = [0.1, 0.3, 0.5, 0.7, 0.9]
y = [0.1, 0.3, 0.5, 0.7, 0.9]
z = [0.3608, 0.7816, 1.58, 2.16, 4.3]

# 为每个数据点添加标签
for (i, j, k) in zip(x, y, z):
    ax.text(i, j, k, f'{k}')

# 散点图
sc = ax.scatter(x, y, z)

# 添加连线，假设最后一个点是最高点
ax.plot([x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]], color='r')

# 设置轴标签
ax.set_xlabel('α')
ax.set_ylabel('β')
ax.set_zlabel('AUC')

# 显示图形
plt.show()
