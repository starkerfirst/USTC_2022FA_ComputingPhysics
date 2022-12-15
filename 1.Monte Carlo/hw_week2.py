#取a=137，c=187，m=256和x subscript 0=1，用线性同余法产生出三维数组，然后绘出其三维和二维分布图形。需要提交源代码和画图结果。
import numpy as np
import matplotlib.pyplot as plt

a=137
c=187
m=256
x0=1
ramdomlist=[]

for i in range(6000):
    x0=(a * x0 + c) % m
    ramdomlist.append(x0 / m)

plt.scatter(ramdomlist[::2], ramdomlist[1::2])
plt.savefig("2D.jpg", dpi=200)

fig=plt.figure(figsize=(10,10), dpi=200)
ax=plt.axes(projection="3d")
ax.scatter3D(ramdomlist[::3], ramdomlist[1::3], ramdomlist[2::3])
plt.savefig("3D.jpg", dpi=200)