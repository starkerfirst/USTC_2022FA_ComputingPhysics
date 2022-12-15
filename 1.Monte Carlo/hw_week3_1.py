import numpy as np
import matplotlib.pyplot as plt

N = 10000

#第一小问
#理论密度曲线
x = np.linspace(0.0, 10, 50)
y = np.exp(-x) #1-exp(-x)与exp(-x)分布一样

#直接抽样
t = np.random.random(N)
x_r = -np.log(t)

#画图
plt.figure(figsize=(10, 10))
plt.plot(x, y, color='blue', label='Theoretical Distribution')
plt.hist(x_r, bins=60, range=(0, 10), density=True, color='yellow', label='Random variable X distribution')
plt.legend()
#plt.show()
#plt.savefig("第三周\\1_(1).jpg")


#第二小问
N = 10000000
#直接抽样
t = np.random.random(N)
x_r = -np.log(t)

fx = x_r**(3/2)
I = sum(fx)/N
print(f"结果：{I}")
I0 = np.sqrt(np.pi)*3/4
print(f"误差：{abs(I0-I)/I0:.4%}")







