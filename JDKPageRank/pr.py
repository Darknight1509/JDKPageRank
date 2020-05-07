import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
alpha=0.85 #系数
N=6434#节点个数
eps=1e-10#后面用到的误差
P=np.zeros([N,N],dtype=float) #转移矩阵
f = open('dependency', 'r') #打开文件dependency，里面是网络的边的情况
while 1:
    line = f.readline()
    if not line:
        break
    pass
    list = line.split()
    P[int(list[0])-1][int(list[1])-1]=1
f.close()
#计算转移概率矩阵
n1=norm(P, axis=1, ord=1)
for i in range(N):
        if(n1[i]==0):
            continue
        P[i,:]=P[i,:]/n1[i]
#计算A矩阵
ee=np.dot(np.ones([N,1]),np.ones([1,N]))
#print(ee)
#print(np.ones([N,N]))

A=alpha*P+(1-alpha)*ee/N

p=np.ones([1,N])*((1+0.0)/N)
print(p)
print(np.ones([1,N])*((1)/N))
pA=np.dot(p,A)
n=0
while(np.linalg.norm(p-pA)>eps):
    p,pA=pA,np.dot(pA,A)
    print(np.linalg.norm(p-pA))
    n=n+1
    print(n)
p1=p*1e7
p1=np.array([p1[0],np.array([i for i in range(N)])])  #加上索引
p1=p1[:,p1[0].argsort()]  #按第一行排序
p1=[p1[0][(N-30):],p1[1][(N-30):]]  #取最大的30个
print(p1)

with open('classname') as file_object:
    contents = file_object.read()
list=contents.split()

s1=np.sum(p1[0])
classname=[] #类名
pr=[]        #比例
for i in range(30):
    index=int(p1[1][i])
    classname.append(list[index])
    pr.append(p1[0][i]/s1)

#打印结果
print("名次"+"    "+"类名"+"    "+"比例")
for i in range(30):
    print(str(i+1)+"    "+classname[29-i]+"    "+str(pr[29-i]))
#画比例图
plt.plot(pr)
plt.show()





