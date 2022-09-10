import util
import matplotlib.pyplot as plt

# execute Sorting Algorithm
for i in range(10,31):
    util.test_time_1000000(i)
    util.test_time(i)

# 實驗結果紀錄
dutch_flag_average=[0.00625,0.0109375,0.0203125,0.0453125,0.09375,0.178125,0.375,0.7609375,1.4890625,2.9078125,5.74375,11.575,24.075,46.253125,95.80625]
lomuto_average=[0.0015625,0.009375,0.0171875,0.0421875,0.1125,0.3140625,1.0265625,3.653125,13.3,51.01875,201.5859375]
hoare_average=[0.0046875,0.00625,0.0140625,0.028125,0.0640625,0.1328125,0.2828125,0.59375,1.2328125,2.5890625,5.4484375,11.634375,24.184375,50.4015625,105.98125]
merge_average=[0.0015625,0.00625,0.021875,0.0453125,0.0953125,0.1984375,0.421875,0.9171875,1.890625,3.95625,8.365625,18.5578125,37.925,79.5046875,167.7046875]
heap_average=[0.0078125,0.0171875,0.0359375,0.0734375,0.171875,0.3609375,0.78125,1.709375,3.5328125,7.5265625,15.9484375,35.1640625,72.8546875,150.340625,319.165625]

arr_size=[i for i in range (10,25)]
lomuto_arr_size=[i for i in range(10,21)]

# generate original plot
plt.figure(figsize=(10,10))
plt.plot(arr_size,dutch_flag_average,color="r",marker="o",label="Quick Sort(Dutch Flag Partition)")
plt.plot(lomuto_arr_size,lomuto_average,color="yellow",marker="o",label="Quick Sort(Lomuto Partition)")
plt.plot(arr_size,hoare_average,color="g",marker="o",label="Quick Sort(Hoare Partition)")
plt.plot(arr_size,merge_average,color="c",marker="o",label="Merge Sort")
plt.plot(arr_size,heap_average,color="m",marker="o",label="Heap Sort")
plt.title("Average Execution Time of Different Sorting Algorithms")
plt.xlabel("Array Size")
plt.ylabel("Time(second)")
plt.legend()

# prediction