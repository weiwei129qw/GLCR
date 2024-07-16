import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

# 示例数据
x = np.array([0.1, 0.1, 0.1, 0.1,0.1,    0.3, 0.3,0.3,0.3,  0.3,    0.5,  0.5,   0.5,0.5,0.5,  0.7, 0.7, 0.7,0.7,0.7, 0.9,0.9,0.9,0.9,0.9])
y = np.array([0.1, 0.3, 0.5, 0.7,0.9,    0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7,0.9,0.1, 0.3, 0.5, 0.7, 0.9,0.1, 0.3, 0.5, 0.7, 0.9, ])
#z = np.array([0.40691,  0.406915, 0.406928, 0.40692,   0.406922,        0.406925,  0.40693, 0.407, 0.40699,0.40698,       0.40698,0.40697,0.40695,0.40694,0.40693,   0.40691,  0.4069,0.40691,0.40692, 0.406923,
   #             0.406924, 0.406925, 0.406925, 0.4069245, 0.406924])

z = np.array([0.4039,  0.404, 0.4042, 0.4045,   0.405,        0.406,  0.406, 0.407, 0.4065,0.406,       0.4058,0.4055,0.4053,0.4051,0.405,   0.4048,  0.4046,0.4045,0.4047, 0.4048,
                0.4049, 0.405, 0.4055, 0.4053, 0.4051])
# 创建网格数据
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制表面
surf = ax.plot_surface(xi, yi, zi, cmap=plt.cm.viridis, edgecolor='none')

# 找到最大值点
max_index = np.argmax(z)
max_x = x[max_index]
max_y = y[max_index]
max_z = z[max_index]

# 使用红色点标记最高点
# 使用更大的点和边框标记最高点
#ax.scatter(max_x, max_y, max_z, color='red', s=100, edgecolors='w', linewidths=2, depthshade = False)  # 增大点的大小，添加白色边框

offset = 0.0008

# 在最高点位置添加注释文本
ax.text(max_x + offset, max_y, max_z + offset, f'α={max_x:.1f}\nβ={max_y:.1f}\nRecall={max_z:.3f}', 
        color='black', va='center', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'), fontsize=8, ha='left')



# 添加颜色条
#fig.colorbar(surf, shrink=0.5, aspect=5)

# 设置轴标签
ax.set_xlabel('α')
ax.set_ylabel('β')
ax.set_zlabel('Recall', rotation=90, labelpad=10)



# 显示图形
plt.show()

