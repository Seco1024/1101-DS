import sys
import numpy as np
import time
import copy

sys.setrecursionlimit(1000000)
    
def printArray(arr, size):
    for i in range(size):
        print(arr[i], end = " ")

def swap (A, i, j):
    temp = A[i]
    A[i] = A[j]
    A[j] = temp
    
# MergeSort definition
def mergeSort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        
        mergeSort(L)
        mergeSort(R)
  
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
  
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
            
# QuickSort(Lomuto) definition
def lomuto_partition(arr, low, high):  
    pivot = arr[high]
    i = (low - 1)  
    for j in range(low, high):
        if (arr[j] <= pivot):
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return (i + 1)


def lomuto_quickSort(arr, low, high):
    if (low < high):
        pi = lomuto_partition(arr, low, high)
        lomuto_quickSort(arr, low, pi - 1)
        lomuto_quickSort(arr, pi + 1, high)
        
# QuickSort(Hoare) definition        
def hoare_partition(arr, low, high):
     
    pivot = arr[low]
    i = low - 1
    j = high + 1
 
    while (True):
        i += 1
        while (arr[i] < pivot):
            i += 1
        j -= 1
        while (arr[j] > pivot):
            j -= 1
 
        if (i >= j):
            return j
        arr[i], arr[j] = arr[j], arr[i]
 
 
def hoare_quickSort(arr, low, high):
    if (low < high):
        pi = hoare_partition(arr, low, high)
        hoare_quickSort(arr, low, pi)
        hoare_quickSort(arr, pi + 1, high)
        
# QuickSort(Dutch Flag) definition
def dutch_flag_partition(A, start, end):
     
    mid = start
    pivot = A[end]
 
    while mid <= end:
        if A[mid] < pivot:
            swap(A, start, mid)
            start += 1
            mid += 1
        elif A[mid] > pivot:
            swap(A, mid, end)
            end -= 1
        else:
            mid += 1
 
    return start - 1, mid

def dutch_flag_quicksort(A, start, end):

    if start >= end:
        return
    if start - end == 1:
        if A[start] < A[end]:
            swap(A, start, end)
        return

    x, y = dutch_flag_partition(A, start, end)
    dutch_flag_quicksort(A, start, x)
    dutch_flag_quicksort(A, y, end)
    
# HeapSort definition
def heapify(arr, n, i):
    largest = i 
    l = 2 * i + 1    
    r = 2 * i + 2    
 
    if l < n and arr[largest] < arr[l]:
        largest = l
 
    if r < n and arr[largest] < arr[r]:
        largest = r
 
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i] 

        heapify(arr, n, largest)

def heapSort(arr):
    n = len(arr)

    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i] 
        heapify(arr, i, 0)
    
dutch_flag_average,lomuto_average,hoare_average,merge_average,heap_average=[[] for y in range(5)]

# test sorting time
def test_time(x):
    dutch_flag_temp,lomuto_temp,hoare_temp,merge_temp,heap_temp=[[] for y in range(5)]
    
    for i in range(1,11):
        temp_arr=np.random.randint(1,1001,2**x)
        dutch_flag_arr,lomuto_arr,hoare_arr,heap_arr,merge_arr=[copy.copy(temp_arr) for i in range(5)]
    
        start=time.process_time()
        dutch_flag_quicksort(dutch_flag_arr,0,len(dutch_flag_arr)-1)
        end=time.process_time()
        dutch_flag_temp.append((end-start))
        
        start=time.process_time()
        lomuto_quickSort(lomuto_arr,0,len(lomuto_arr)-1)
        end=time.process_time()
        lomuto_temp.append((end-start))
        
        start=time.process_time()
        hoare_quickSort(hoare_arr,0,len(hoare_arr)-1)
        end=time.process_time()
        hoare_temp.append((end-start))
        
        start=time.process_time()
        mergeSort(merge_arr)
        end=time.process_time()
        merge_temp.append((end-start))
        
        start=time.process_time()
        heapSort(heap_arr)
        end=time.process_time()
        heap_temp.append((end-start))

    sorting_list=["dutch_flag","lomuto","hoare","merge","heap"]
    for i in sorting_list:
        exec(f"{i}_average.append(sum({i}_temp)/len({i}_temp))")
        print(f"{i}處理2的{x}次方筆資料的平均執行時間：",eval(f"{i}_average[{x}-10]"),"秒")
        
dutch_flag_average,lomuto_average,hoare_average,merge_average,heap_average=[[] for y in range(5)]

def test_time_1000000(x):
    dutch_flag_temp,lomuto_temp,hoare_temp,merge_temp,heap_temp=[[] for y in range(5)]
    
    for i in range(1,11):
        temp_arr=np.random.randint(1,100,2**x)
        dutch_flag_arr,lomuto_arr,hoare_arr,heap_arr,merge_arr=[copy.copy(temp_arr) for i in range(5)]
    
        start=time.process_time()
        dutch_flag_quicksort(dutch_flag_arr,0,len(dutch_flag_arr)-1)
        end=time.process_time()
        dutch_flag_temp.append((end-start))
        
        start=time.process_time()
        lomuto_quickSort(lomuto_arr,0,len(lomuto_arr)-1)
        end=time.process_time()
        lomuto_temp.append((end-start))
        
        start=time.process_time()
        hoare_quickSort(hoare_arr,0,len(hoare_arr)-1)
        end=time.process_time()
        hoare_temp.append((end-start))
        
        start=time.process_time()
        mergeSort(merge_arr)
        end=time.process_time()
        merge_temp.append((end-start))
        
        start=time.process_time()
        heapSort(heap_arr)
        end=time.process_time()
        heap_temp.append((end-start))

    sorting_list=["dutch_flag","lomuto","hoare","merge","heap"]
    for i in sorting_list:
        exec(f"{i}_average.append(sum({i}_temp)/len({i}_temp))")
        print(f"{i}處理2的{x}次方筆資料的平均執行時間：",eval(f"{i}_average[{x}-10]"),"秒")