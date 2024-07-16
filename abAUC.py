import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 示例数据
x = [0.1, 0.1, 0.3, 0.5, 0.3, 0.5,0.7,0.9,0.7, 0.9]
y = [0.1, 0.3, 0.3, 0.3, 0.5, 0.7,0.7,0.9,0.9, 0.9]
z = [0.501, 0.52,0.529, 0.545, 0.782, 0.551,0.565,0.6, 0.754,0.609]

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 散点图
sc = ax.scatter(x, y, z)

# 为每个数据点添加标签
for (i, j, k) in zip(x, y, z):
    ax.text(i, j, k, f'{k:.3f}')

# 找到最大和最小的Z值对应的索引
min_z_idx = np.argmin(z)
max_z_idx = np.argmax(z)

# 添加从最小值到最大值的连线
ax.plot([x[min_z_idx], x[max_z_idx]], [y[min_z_idx], y[max_z_idx]], [z[min_z_idx], z[max_z_idx]], color='r')
# 添加从最大值到z的最后一个值的连线
ax.plot([x[max_z_idx], x[-1]], [y[max_z_idx], y[-1]], [z[max_z_idx], z[-1]], color='b', linestyle='--')
# 在线段的末端（x[-1], y[-1], z[-1]）添加一个较大的点作为“箭头”
ax.scatter([x[-1]], [y[-1]], [z[-1]], color='b')  # s控制点的大小
# 设置轴标签
ax.set_xlabel('α')
ax.set_ylabel('β')
ax.set_zlabel('AUC', rotation=90)

# 显示图形
plt.show()