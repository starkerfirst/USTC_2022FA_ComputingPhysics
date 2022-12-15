# A并不需要知道，因为在分子分母中会被消去
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import time

# 生成器部分代码：
rng = default_rng()

def r_fx_fy(x, y):
    return np.exp(-(x**2 - y**2)/2)

# 总步数
N = 1000000
# 每一步的步长范围
deltalist = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
# 接受点与试探步数之比
ratio = []
# 到达平衡分布的时间
timelist = []

for delta in deltalist:
    # 方差稳定检验变量
    test = 0
    loop = 0
    #记录步数
    n_accept = 0
    # 起点
    x0 = 0
    walk_path = [x0]
    a = time.time()
    while(True):
        for i in range(N):
            # 取一个试探位置
            x_try = rng.uniform(x0-delta, x0+delta)
            # 计算r，判断是否接受
            r = r_fx_fy(x_try, x0)
            if r > 1:
                x0 = x_try
                walk_path.append(x0)
                n_accept += 1
                continue
            if r >= rng.random():
                x0 = x_try
                walk_path.append(x0)
                n_accept += 1
                continue
            walk_path.append(x0)
        var = np.var(np.array(walk_path))
        # 方差稳定性检验
        if( var > 0.95 and var < 1.05):  test += 1 
        loop += 1
        if(test == 4): break
    
    b = time.time()
    print(f"n_accept/N={n_accept/N/loop:.4},var={var}")
    ratio.append(n_accept/N/loop)
    timelist.append(b-a)

# 理论密度曲线
def guass(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)

x = np.linspace(-10, 10, 200)
y = guass(x)

plt.figure(figsize=(5, 5), dpi=100)
plt.hist(walk_path, bins=60, range=(-7.0, 7.0), density=True, color='yellow', label='walk_path distribution')
plt.plot(x, y, color='blue', label='Theoretical Distribution')
plt.legend(loc='upper right')
plt.show()



# 接受点与试探步数之比，到达平衡分布的时间与最大试探步长𝛿的关系作图

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(deltalist, ratio, color='blue', label='Acceptance Ratio')
#plt.plot(deltalist, timelist, color='green', label='Time')
plt.legend(loc='upper left')
plt.xlabel('delta')
plt.ylabel('ratio')
plt.show()

plt.figure(figsize=(5, 5), dpi=100)
#plt.plot(deltalist, ratio, color='blue', label='Acceptance Ratio')
plt.plot(deltalist, timelist, color='green', label='Time')
plt.legend(loc='upper left')
plt.xlabel('delta')
plt.ylabel('time/s')
plt.show()

# 可以发现步长在4左右收敛速度最快