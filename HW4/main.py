import util
import matplotlib.pyplot as plt

# 鄉村數量實驗
p=[]
def rshow(n):
    g=util.generate_graph(n)
    print("節點數量：",g.n)
    print("鄉村數量：",int(len(g.v_rural)))
    ans=int(len(g.v_rural))/g.n
    print("鄉村數量佔總體比例：",ans)
    p.append(ans)
    
for i in range(20,410,20):
    util.rshow(i)
print(p)

# 鄉村數量折線圖
size=[i for i in range(20,410,20)]
plt.figure(figsize=(10,10))
plt.plot(size,p)
plt.xticks(range(20,410,20))
plt.xlabel("n")
plt.ylabel("numbers")
plt.title("proportion of rural")

# 主要城市實驗
def megan(n):
    g=util.generate_graph(n)
    return int(len(g.v_mega)/2)

meganum=[]
for i in range(2,41,2):
    meganum.append(megan(10*i))
print(meganum)

# 主要城市數量折線圖
size=[i for i in range(20,410,20)]
plt.figure(figsize=(10,10))
plt.plot(size,meganum)
plt.xticks(range(20,410,20))
plt.xlabel("n")
plt.ylabel("numbers")
plt.title("# maga cities")

# 生成網路圖
for i in range(10):
    util.show(100)
    


# 第二題
def main(n):
    total=0
    for i in range(50):
        total+=util.average(n)
    return total/50

average_record=[]
for i in range(2,41,2):
    average_record.append(main(10*i))
print(average_record)

size=[i for i in range(20,410,20)]
plt.figure(figsize=(10,10))
plt.plot(size,average_record)
plt.xticks(range(20,410,20))
plt.xlabel("n")
plt.ylabel("Distance")
plt.title("Average distance")

# 第三題
a=util.generate_graph(100)
b=a.AdjacencyList
new=[]

for i in b:
    new.append(b[i])
    
for i in new:
    util.fdijkstra(new,1,sink=None)
    
print(new)

