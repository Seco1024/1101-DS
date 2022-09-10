import util
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

with open("result_1_dat",mode="w") as file_1: #建立 insertion的 raw data 檔案
    file_1.write('''2^k  Hash Table  Treap  Skip List  Sorted Array \n''') #寫入 raw data 檔案欄位名
with open("result_2_dat",mode="w") as file_2: #建立 search 的 raw data 檔案
    file_2.write('''2^k  Hash Table  Treap  Skip List  Sorted Array \n''') #寫入 raw data 檔案欄位名

def main(x): #實驗函式

    global result_1_dat,result_2_dat,cul #global data

    with open("result_1_dat",mode="a") as file_1: #開啟 insertion 的 raw data 檔案
            file_1.write(f"{x}   ") #寫入 raw data列名
    with open("result_2_dat",mode="a") as file_2: #開啟 search 的 raw data 檔案
            file_2.write(f"{x}   ") #寫入 raw data列名

    #HashTable
    h=util.HashTable(2**x) #建立空資料結構
    start=time.process_time() #開始計時
    for i in range(2**x): #迴圈：插入2^k筆隨機資料
        h.insert(random.randint(1,2**30)) 
    end=time.process_time() #結束計時
    print(f"Hash Table插入:{x}次方花費{end-start}秒") #印出結果
    
    start_s=time.process_time() #開始計時
    for i in range(100000): #迴圈：搜尋十萬筆筆隨機資料
        h.search(random.randint(1,2**30)) 
    end_s=time.process_time() #結束計時
    print(f"Hash Table搜尋:{x}次方花費{end_s-start_s}秒") #印出結果
    
    with open("result_1_dat",mode="a") as file_1: #開啟insertion檔案
        file_1.write(f"{end-start}    ") #寫入結果
        file_1.close() #關閉檔案
    
    with open("result_2_dat",mode="a") as file_2: #開啟search檔案
        file_2.write(f"{end_s-start_s}    ") #寫入結果
        file_2.close() #關閉檔案
    
    
    #Treap
    
    root = None
    start=time.process_time()
    for i in range(2**x):
        root=util.insertNode(root,random.randint(1,2**30))
    end=time.process_time()
    print(f"Treap插入:{x}次方花費{end-start}秒")
    
    start_s=time.process_time()
    for i in range(100000):
        util.searchNode(root,random.randint(1,2**30))
    end_s=time.process_time()
    print(f"Treap搜尋:{x}次方花費{end_s-start_s}秒")
    
    with open("result_1_dat",mode="a") as file_1:
        file_1.write(f"{end-start}    ")
        file_1.close()
    result_1_dat[(x-10)].append(end-start)
    
    with open("result_2_dat",mode="a") as file_2:
        file_2.write(f"{end_s-start_s}    ")
        file_2.close()
    result_2_dat[(x-10)].append(end_s-start_s)
    
    
    #SkipList
    
    lst=util.SkipList(1000,0.5)
    start=time.process_time()
    for i in range(2**x):
        lst.insertElement(random.randint(1,2**30))
    end=time.process_time()
    print(f"Skip List插入:{x}次方花費{end-start}秒")
    
    start_s=time.process_time()
    for i in range(100000):
        lst.searchElement(random.randint(1,2**30))
    end_s=time.process_time()
    print(f"Skip List搜尋:{x}次方花費{end_s-start_s}秒")
    
    with open("result_1_dat",mode="a") as file_1:
        file_1.write(f"{end-start}    ")
        file_1.close()
    result_1_dat[(x-10)].append(end-start)
    
    with open("result_2_dat",mode="a") as file_2:
        file_2.write(f"{end_s-start_s}    ")
        file_2.close()
    result_2_dat[(x-10)].append(end_s-start_s)
    
    
    #Sorted Array
    
    arr=util.SortedArray()
    start=time.process_time()
    for i in range(2**x):
        arr.insertion(random.randint(1,2**30)) 
    end=time.process_time()
    print(f"Sorted Array插入:{x}次方花費{end-start}秒")
    
    start_s=time.process_time()
    for i in range(100000):
        j=random.randint(1,2**30)
        arr.binary_search(arr,j)
    end_s=time.process_time()
    print(f"Sorted Array搜尋:{x}次方花費{end_s-start_s}秒")
    
    
    with open("result_1_dat",mode="a") as file_1:
        file_1.write(f"{end-start}    \n")
        file_1.close()
    result_1_dat[(x-10)].append(end-start)
    
    with open("result_2_dat",mode="a") as file_2:
        file_2.write(f"{end_s-start_s}    \n")
        file_2.close()
    result_2_dat[(x-10)].append(end_s-start_s)
    
# 執行實驗
for i in range(10,31):
    main(i)
    
arr=[x for x in range(10,25)]
arr2=[x for x in range(10,17)]

## 創建數據集
plt.scatter(arr2,util.sortedarray_s)
plt.show()

## 組合成DataFrame格式
data_dict = {'arr':arr2,'time':util.sortedarray_s}
df = pd.DataFrame(data_dict)
X = df[['arr']]
y = df[['time']]

## 訓練多項式迴歸模型
regressor = make_pipeline(PolynomialFeatures(2), LinearRegression())
regressor.fit(X,y)

arr_0=[24,25,26,27,28,29,30]
arr_00=[x for x in range(16,31)]
ddata_dict={'arrr':arr_00}
df=pd.DataFrame(ddata_dict)
X_test=df[['arrr']]

print(X_test)

y_pred=regressor.predict(X_test)
print(y_pred)

## 視覺化
plt.scatter(X,y)
plt.plot(X, regressor.predict(X), color = 'blue')
plt.show()


hashtable_i=[0.015625,0.03125,0.046875,0.09375,0.265625,0.453125,0.90625,
           1.921875,3.890625,7.609375,15.765625,31.40625,60.09375,124.109375,
           303.5625]
treap_i=[0.015625,0.0,0.03125,0.078125,0.125,0.28125,0.65625,1.25,
         2.78125,7.0,14.03125,34.8125,73.171875,166.734375,682.609375]
skiplist_i=[0.015625,0.015625,0.03125,0.078125, 0.234375,0.390625,0.8125,
            2.09375,4.4375,10.203125,20.8125,46.546875,102.328125,322.859375,
            1083.96875]
sortedarray_i=[0.140625,0.546875,2.171875,8.90625,36.015625,140.859375,
               571.203125]

hashtable_s=[0.765625,0.765625,0.78125,0.78125,0.859375,0.828125,0.875,0.828125,
             0.828125,0.84375,0.84375,0.890625,0.859375,0.828125,0.875]
treap_s=[0.28125,0.328125,0.34375,0.375,0.4375,0.4375,0.578125,0.609375,
         0.703125,0.796875,0.890625,1.0,1.140625,1.328125,1.53125]
skiplist_s=[0.3125,0.359375,0.40625,0.4375,0.5,0.515625,0.59375,0.734375,
            0.84375,0.953125,1.078125,1.296875,1.375,1.59375,2.015625]
sortedarray_s=[0.09375,0.09375,0.09375,0.09375,0.07815,0.0625,0.09375]

#prediction
hashtable_ip=[303.5625,576.05205482,1028.73956583,1723.46876858,2743.23621982,
             4187.56545835,6173.96631203]

hashtable_sp=[0.875,0.88467262,0.89148065,0.89828869,0.90509673,0.91190476,0.9187128]

treap_ip=[682.609375,1441.92015449,2781.86765675,4903.13493711,8087.26934096,12673.65600834,19064.73335317]

treap_sp=[1.53125,1.65728022,1.84181834,2.03821913,2.24648261,2.46660876,2.69859759]

skiplist_ip=[1083.96875,2275.27484175,4335.58309827,7575.14206458,12413.29493032,19354.69207083,28996.92648519]

skiplist_sp=[2.015625,2.12537775,2.37936985,2.65044138,2.93859234,3.24382272,3.56613253]

sortedarray_ip=[571.203125,1812.73658692,4699.58811369,10496.84064969,20991.44730449,38581.619707,66366.34363026,108234.89461624,
                168956.35360065,254269.12253771,370970.44002494,527005.89692799,731558.95200541,995140.44753343,
                1329678.12493079]
sortedarray_sp=[0.09375,0.08258571,0.08481071,0.08815,0.09260357,0.09817143,0.10485357,0.11265,0.12156071,0.13158571,0.142725,
                0.15497857,0.16834643,0.18282857,0.198425]

# 畫圖
## with pred. insertion
plt.figure(figsize=(10,10))
plt.plot(arr,hashtable_s,color="r",marker="o",label="Hash Table")
plt.plot(arr,treap_s,color="darkorange",marker="o",label="Treap")
plt.plot(arr,skiplist_s,color="g",marker="o",label="Skip List")
plt.plot(arr2,sortedarray_s,color="c",marker="o",label="Sorted Array")

plt.plot(arr_0,hashtable_sp,color="r",marker='o',label="Hash Table pred.",linestyle='--')
plt.plot(arr_0,treap_sp,color="darkorange",marker='o',label="Treap pred.",linestyle='--')
plt.plot(arr_0,skiplist_sp,color="g",marker='o',label="Skip List pred.",linestyle='--')
plt.plot(arr_00,sortedarray_sp,color="c",marker='o',label="Sorted Array pred.",linestyle='--')

plt.title("Search Time of Different Data Structures")
plt.xlabel("Array Size(2^k)")
plt.ylabel("Time")
plt.xticks(list(range(10,31,1)))
plt.legend()
## with no pred. search
plt.figure(figsize=(10,10))
plt.plot(arr,hashtable_s,color="r",marker="o",label="Hash Table")
plt.plot(arr,treap_s,color="darkorange",marker="o",label="Treap")
plt.plot(arr,skiplist_s,color="g",marker="o",label="Skip List")
plt.plot(arr2,sortedarray_s,color="c",marker="o",label="Sorted Array")

plt.title("Insertion Time of Different Data Structures with no pred.")
plt.xlabel("Array Size(2^k)")
plt.ylabel("Time")
plt.xticks(list(range(10,25,1)))
plt.legend()

    
    
    