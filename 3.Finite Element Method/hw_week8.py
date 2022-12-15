import numpy as np
import matplotlib.pyplot as plt

#第一步：初始化

# 求解区域的范围
x_range = [0.0, 1.0]
y_range = [0.0, 1.0]

# 划分的步长
h = 0.01

# 每一个点的坐标
x = np.arange(x_range[0], x_range[1]+h, h)
y = np.arange(y_range[0], y_range[1]+h, h)

# 格点数目
x_n_grid = len(x)
y_n_grid = len(y)

# 边界点和求解区域，边界点标记为True，待求点标记为False
area = np.full(shape=(x_n_grid, y_n_grid), fill_value=True)
area[1: -1, 1: -1] = np.full(shape=(x_n_grid-2, y_n_grid-2), fill_value=False)
area_tri = np.tril(area)
for i in range(x_n_grid):
    area_tri[i, x_n_grid-i-1] = True

# 建立编号矩阵，其中的每一个编号对应区域中的一个节点
index_matrix = np.zeros((x_n_grid, y_n_grid), dtype=int)
k = 0

# 先对内部节点编号
for i in range(x_n_grid):
    for j in range(y_n_grid):
        if not area[i, j]:
            index_matrix[i, j] = k
            k += 1
            if i == j:
                continue
            
# 内部节点数            
n_inner = k

# 然后对边界节点编号
for i in range(x_n_grid):
    for j in range(y_n_grid):
        if area[i, j]:
            index_matrix[i, j] = k
            k += 1
            if i == j:
                continue


# 边界节点数        
n_edge = k - n_inner

# 第二步：划分区域
class element:
    def __init__(self, index_list, x_list, y_list):
        # 记录三角形元素的三个节点编号和坐标
        self.index = index_list
        self.x = x_list
        self.y = y_list
        # 计算三角形元素的b和c
        self.b = [self.y[(i+1)%3]-self.y[(i+2)%3] for i in range(3)]
        self.c = [self.x[(i+2)%3]-self.x[(i+1)%3] for i in range(3)]

e_list = []

for i in range(x_n_grid-1):
    for j in range(1, y_n_grid): #（i,j）是固定参考点
        if(i >= j):
            e_list.append(
                element(
                    index_list=[index_matrix[i, j], index_matrix[i+1, j-1], index_matrix[i, j-1]],
                    x_list=[x[i], x[i+1], x[i]],
                    y_list=[y[j], y[j-1], y[j-1]]
                )
            )
        if(i == j):
            e_list.append(
                element(
                    index_list=[index_matrix[i, j], index_matrix[i+1, j], index_matrix[i+1, j-1]],
                    x_list=[x[i], x[i+1], x[i+1]],
                    y_list=[y[j], y[j], y[j-1]]
                )
            )

# 第三步：建立矩阵

Delta = (x_range[1] - x_range[0])*(y_range[1] - y_range[0])/len(e_list)/2

K = np.zeros((n_inner + n_edge, n_inner + n_edge))

# K矩阵是对称的，所以只计算一半
for e in e_list:
    for i in range(3):
        for j in range(i, 3):
            K[e.index[i], e.index[j]] += (e.b[i]*e.b[j] + e.c[i]*e.c[j])/4/Delta

# 根据对称性补全K矩阵
K += K.T - np.diag(np.diag(K))



# 第四步：SOR迭代
# 记录k_ij不为0的编号
k_ij_index = []
for i in range(n_inner):
    k_ij_index.append([]) # k_ij_index[i] 指第i行的非零元素
    for j in range(n_inner+n_edge):
        if K[i, j] != 0 and i != j:
            k_ij_index[-1].append(j)


omega = 1.5

# 在迭代公式中，使用行向量更加简单（因为相当于一位数组，指标更简单）
# 使用行向量记录Phi的每一个值
# 采用K矩阵，但是不更新边界点的值（等效于A）
X = np.hstack((np.zeros(n_inner), Phi_2.T[0]))
X0 = X.copy()

# 迭代时只改变边界节点
for i in range(n_inner):
    X[i] = (1-omega)*X[i]
    a = omega/K[i, i]
    for j in k_ij_index[i]:
        X[i] -= a*K[i, j]*X[j]

while np.max(abs(X-X0)) > 1e-6:
    X0 = X.copy()
    for i in range(n_inner):
        X[i] = (1-omega)*X[i]
        a = omega/K[i, i]
        for j in k_ij_index[i]:
            X[i] -= a*K[i, j]*X[j]
    

Phi = X.copy()

value_matrix = np.zeros((x_n_grid, y_n_grid))

# 将解填入求解区域中
for i in range(x_n_grid):
    for j in range(y_n_grid):
        if(i>=j):
            value_matrix[i, j] = Phi[index_matrix[i, j]]
            
fig = plt.figure(figsize=(5, 5), dpi=100)
plt.contourf(x, y, value_matrix.T)
plt.show()