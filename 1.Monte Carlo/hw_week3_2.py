import numpy as np
import matplotlib.pyplot as plt

#第一小问
#理论密度曲线
x = np.linspace(-10, 10, 200)
y = 1/(np.pi*(x**2+1))

#直接抽样
t = np.random.random(10000)
x_r = np.tan(np.pi*(t-0.5))

#画图
plt.figure(figsize=(10, 10))
plt.plot(x, y, color='blue', label='Theoretical Distribution')
plt.hist(x_r, bins=150, range=(-10, 10), density=True, color='yellow', label='Random variable X distribution')
plt.legend()
#plt.show()
#plt.savefig("第三周\\2_(1).jpg")


#第二小问

N = 100000000
#直接抽样
t = np.random.random(N)
x_r = np.tan(np.pi*t/2)

#计算积分的估计值
y = np.pi*np.sqrt(x_r)/2
I = sum(y)/N
print(f"结果：{I}")
I0 = np.pi/np.sqrt(2)
print(f"误差：{abs(I0-I)/I0:.4%}")
