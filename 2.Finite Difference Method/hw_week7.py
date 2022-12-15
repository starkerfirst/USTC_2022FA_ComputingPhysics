import numpy as np
import matplotlib.pyplot as plt

# 求解区域的范围
x_range = [0.0, np.pi]
y_range = [0.0, np.pi]

# 划分的步长
h = np.pi / 90

# 每一个点的坐标
x = np.arange(x_range[0], x_range[1]+h, h)
y = np.arange(y_range[0], y_range[1]+h, h)

# 格点数目
x_n_grid = len(x)
y_n_grid = len(y)

# 边界点和求解区域，边界点标记为True，待求点标记为False
area = np.full(shape=(x_n_grid, y_n_grid), fill_value=True)
area[1: -1, 1: -1] = np.full(shape=(x_n_grid-2, y_n_grid-2), fill_value=False)

# 建立编号矩阵，其中的每一个编号对应区域中的一个点
index_matrix = np.zeros((x_n_grid, y_n_grid), dtype=int)
k = 0

for i in range(x_n_grid):
    for j in range(y_n_grid):
        # 如果对应的是边界点，则记为-1
        if area[i, j]:
            index_matrix[i, j] = -1
        # 如果对应的是待求点，则依次编号
        else:
            index_matrix[i, j] = k
            k += 1
            
# 待求点的数量
size = k 

# 边界条件
def edge_value(x, y):
    if x == 0.0 or x == np.pi or y == 0.0:
        return 0.0
    if y == np.pi :
        return np.sin(x)

# 建立函数值矩阵
value_matrix = np.zeros(shape=(x_n_grid, y_n_grid))

# 填入边界值
for i in range(x_n_grid):
    for j in range(y_n_grid):
        if area[i, j]:
            value_matrix[i, j] = edge_value(x[i], y[j])
            
# q(x,y)
def q(x, y):
    return 0.0

# 计算每一点处的q(x,y)
q_matrix = np.zeros((x_n_grid, y_n_grid))

for i in range(x_n_grid):
    for j in range(y_n_grid):
        if not area[i, j]:
            q_matrix[i, j] = q(x[i], y[j])

# 建立列表用来记录系数矩阵和常数向量
A = []
b = []

for i in range(x_n_grid):
    for j in range(y_n_grid):
        # 如果是待求点
        if not area[i, j]:
            # 初始化Aij和bij
            Aij = np.zeros(size)
            bij = -0.25*h*h*q_matrix[i, j]
            # Aij的对应位置为1
            Aij[index_matrix[i, j]] = 1.0
            # 分别判断周围的点是待求点还是边界点
            # 如果是边界点，bij的对应位置为1/4*value
            # 如果是待求点，Aij的对应位置为-1/4
            if area[i-1, j]:
                bij += 0.25*value_matrix[i-1, j]
            else:
                Aij[index_matrix[i-1, j]] =-0.25
            if area[i+1, j]:
                bij += 0.25*value_matrix[i+1, j]
            else:
                Aij[index_matrix[i+1, j]] =-0.25
            if area[i, j-1]:
                bij += 0.25*value_matrix[i, j-1]
            else:
                Aij[index_matrix[i, j-1]] =-0.25
            if area[i, j+1]:
                bij += 0.25*value_matrix[i, j+1]
            else:
                Aij[index_matrix[i, j+1]] =-0.25
            # 将Aij和bij加入A和b
            A.append(Aij)
            # 写成[bij]是因为b是一个列向量
            b.append([bij])

# 将A和b转换为ndarray格式，以便进行矩阵运算
A = np.array(A)
b = np.array(b)

#输出A
fig = plt.figure(figsize=(10,10), dpi=100)
plt.imshow(A)
plt.show()


# SOR法解方程
# 确定omega
omega = 7/4.0 

# 迭代需要的各个矩阵和向量
I = np.diag(np.ones(size))
D = np.diag(np.diag(A))
U = np.triu(A, k=1)
L = np.tril(A, k=-1)
D_inv = np.linalg.inv(np.diag(np.diag(A)))
tmp = np.linalg.inv(I+omega*D_inv.dot(L))
S_omega = tmp.dot((1-omega)*I-omega*D_inv.dot(U))
f_omega = omega*tmp.dot(D_inv.dot(b))

# 迭代初始值
X0 = np.zeros((size, 1))

X = S_omega.dot(X0) + f_omega

# 迭代次数
k = 1

while np.max(abs(X-X0)) > 1e-5:
    X0 = X
    X = S_omega.dot(X0) + f_omega
    k += 1

#将解填入求解区域中
for i in range(x_n_grid):
    for j in range(y_n_grid):
        if not area[i, j]:
            value_matrix[i, j] = X[index_matrix[i, j], 0]
            
fig = plt.figure(figsize=(5, 5), dpi=100)
plt.contourf(x, y, value_matrix.T)
plt.colorbar()
plt.show()

# 理论解
# 建立理论值矩阵
theoretical_matrix = np.zeros(shape=(x_n_grid, y_n_grid))
for i in range(x_n_grid):
    for j in range(y_n_grid):
        theoretical_matrix[i, j] = np.sin(x[i]) * np.sinh(y[j]) / np.sinh(np.pi)
        

fig = plt.figure(figsize=(5, 5), dpi=100)
plt.contourf(x, y, theoretical_matrix.T)
plt.colorbar()
plt.show()

""" #差值矩阵建立 
diff = theoretical_matrix - value_matrix
fig = plt.figure(figsize=(5, 5), dpi=100)
plt.imshow(diff.T)
plt.colorbar()
plt.show() """