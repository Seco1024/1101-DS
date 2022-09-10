from random import randrange
import random
import numpy as np
import struct

# treap definition
class TreapNode: # A Treap Node
    # constructor
    def __init__(self, data, priority=100, left=None, right=None):
        self.data = data
        self.priority = randrange(priority)
        self.left = left
        self.right = right
 
 
''' Function to left-rotate a given treap
 
      r                       R
     / \     Left Rotate     / \
    L   R       ———>        r   Y
       / \                     / \
      X   Y                   L   X
'''
 

def rotateLeft(root):
 
    R = root.right
    X = root.right.left
 
    # rotate
    R.left = root
    root.right = X
 
    # set a new root
    return R
 

'''Function to right-rotate a given treap
 
        r                        L
       / \     Right Rotate     / \
      L   R       ———>         X   r
     / \                          / \
    X   Y                        Y   R
'''
 

def rotateRight(root):
 
    L = root.left
    Y = root.left.right
 
    # rotate
    L.right = root
    root.left = Y
 
    # set a new root
    return L
 

# Recursive function to insert a given key with a priority into treap
def insertNode(root, data):
 
    # base case
    if root is None:
        return TreapNode(data)
 
    # if the given data is less than the root node, insert in the left subtree;
    # otherwise, insert in the right subtree
    if data < root.data:
        root.left = insertNode(root.left, data)
 
        # rotate right if heap property is violated
        if root.left and root.left.priority > root.priority:
            root = rotateRight(root)
    else:
        root.right = insertNode(root.right, data)
 
        # rotate left if heap property is violated
        if root.right and root.right.priority > root.priority:
            root = rotateLeft(root)
 
    return root
 

#Recursive function to search for a key in a given treap
def searchNode(root, key):
 
    # if the key is not present in the tree
    if root is None:
        return False
 
    # if the key is found
    if root.data == key:
        return True
 
    # if the key is less than the root node, search in the left subtree
    if key < root.data:
        return searchNode(root.left, key)
 
    # otherwise, search in the right subtree
    return searchNode(root.right, key)
 

# Recursive function to delete a key from a given treap
def deleteNode(root, key):
 
    # base case: the key is not found in the tree
    if root is None:
        return None
 
    # if the key is less than the root node, recur for the left subtree
    if key < root.data:
        root.left = deleteNode(root.left, key)
 
    # if the key is more than the root node, recur for the right subtree
    elif key > root.data:
        root.right = deleteNode(root.right, key)
 
    # if the key is found
    else:
 
        # Case 1: node to be deleted has no children (it is a leaf node)
        if root.left is None and root.right is None:
            # deallocate the memory and update root to None
            root = None
 
        # Case 2: node to be deleted has two children
        elif root.left and root.right:
            # if the left child has less priority than the right child
            if root.left.priority < root.right.priority:
                # call `rotateLeft()` on the root
                root = rotateLeft(root)
 
                # recursively delete the left child
                root.left = deleteNode(root.left, key)
            else:
                # call `rotateRight()` on the root
                root = rotateRight(root)
 
                # recursively delete the right child
                root.right = deleteNode(root.right, key)
 
        # Case 3: node to be deleted has only one child
        else:
            # choose a child node
            child = root.left if (root.left) else root.right
            root = child
 
    return root
 
# Utility function to print two-dimensional view of a treap using
# reverse inorder traversal
def printTreap(root, space):
 
    height = 10
 
    # Base case
    if root is None:
        return
 
    # increase distance between levels
    space += height
 
    # print the right child first
    printTreap(root.right, space)
 
    # print the current node after padding with spaces
    for i in range(height, space):
        print(' ', end='')
 
    print((root.data, root.priority))
 
    # print the left child
    printTreap(root.left, space)
    
# Skip List definition
class Node(object):
    '''
    Class to implement node
    '''
    def __init__(self, key, level):
        self.key = key
  
        # list to hold references to node of different level 
        self.forward = [None]*(level+1)

class SkipList(object):
    '''
    Class for Skip list
    '''
    def __init__(self, max_lvl, P):
        # Maximum level for this skip list
        self.MAXLVL = max_lvl
  
        # P is the fraction of the nodes with level 
        # i references also having level i+1 references
        self.P = P
  
        # create header node and initialize key to -1
        self.header = self.createNode(self.MAXLVL, -1)
  
        # current level of skip list
        self.level = 0
      
    # create  new node
    def createNode(self, lvl, key):
        n = Node(key, lvl)
        return n
      
    # create random level for node
    def randomLevel(self):
        lvl = 0
        while random.random()<self.P and \
              lvl<self.MAXLVL:lvl += 1
        return lvl
  
    # insert given key in skip list
    def insertElement(self, key):
        # create update array and initialize it
        update = [None]*(self.MAXLVL+1)
        current = self.header
  
        '''
        start from highest level of skip list
        move the current reference forward while key 
        is greater than key of node next to current
        Otherwise inserted current in update and 
        move one level down and continue search
        '''
        for i in range(self.level, -1, -1):
            while current.forward[i] and \
                  current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
  
        ''' 
        reached level 0 and forward reference to 
        right, which is desired position to 
        insert key.
        ''' 
        current = current.forward[0]
  
        '''
        if current is NULL that means we have reached
           to end of the level or current's key is not equal
           to key to insert that means we have to insert
           node between update[0] and current node
       '''
        if current == None or current.key != key:
            # Generate a random level for node
            rlevel = self.randomLevel()
            global cul,mlvl
            cul+=(rlevel)
            if mlvl<=rlevel:
                mlvl=rlevel
            '''
            If random level is greater than list's current
            level (node with highest level inserted in 
            list so far), initialize update value with reference
            to header for further use
            '''
            if rlevel > self.level:
                for i in range(self.level+1, rlevel+1):
                    update[i] = self.header
                self.level = rlevel
  
            # create new node with random level generated
            n = self.createNode(rlevel, key)
  
            # insert node by rearranging references 
            for i in range(rlevel+1):
                n.forward[i] = update[i].forward[i]
                update[i].forward[i] = n

  
    def searchElement(self, key): 
        current = self.header
  
        '''
        start from highest level of skip list
        move the current reference forward while key 
        is greater than key of node next to current
        Otherwise inserted current in update and 
        move one level down and continue search
        '''
        for i in range(self.level, -1, -1):
            while(current.forward[i] and\
                  current.forward[i].key < key):
                current = current.forward[i]
  
        # reached level 0 and advance reference to 
        # right, which is prssibly our desired node
        current = current.forward[0]
  
        # If current node have key equal to
        # search key, we have found our target node
            
    def displayList(self):
        print("\n*****Skip List******")
        head = self.header
        for lvl in range(self.level+1):
            print("Level {}: ".format(lvl), end=" ")
            node = head.forward[lvl]
            while(node != None):
                print(node.key, end=" ")
                node = node.forward[lvl]
            print("")
            
# sorted array definition
class SortedArray:
    def __init__(self):
        self.list=[]
        self.len=len(self.list)

    def insertion(self,x):
        self.list.append(x)
        for i in range(len(self.list)):
            key=self.list[i]
            j=i-1
            while j>=0 and key<self.list[j]:
                self.list[j+1]=self.list[j]
                j-=1
            self.list[j+1]=key
            
            
    def printing(self):
        print(self.list)
        
            
    def binary_search(self,data,key_bs):
        low=0
        high=data.len-1
        while low <= high:
            mid=int((low+high)/2)
            if key_bs==self.list[mid]:
                return mid
            elif key_bs>self.list[mid]:
                low=mid+1
            else:
                high=mid-1
        return -1
# linked list definition
class node:
    def __init__(self,data=None):
        self.data=data
        self.next=None
        
class LinkedList:
    def __init__(self):
        self.head=node()
    
    def insert_data(self,data):
        new=node(data)
        temp=self.head
        while temp.next!=None:
            temp=temp.next
        temp.next=new
        
    def __str__(self):
        temp_node=self.head
        show=[]
        while temp_node.next!=None:
            temp_node=temp_node.next
            show.append(temp_node.data)
        return str(show)
    
    def __repr__(self):
        temp_node=self.head
        show=[]
        while temp_node.next!=None:
            temp_node=temp_node.next
            show.append(temp_node.data)
        return repr(show)
        
    def length(self):
        total=0
        temp=self.head
        while temp.next!=None:
            total+=1
            temp=temp.next
        return total
    
    def search_data(self,data):
        temp=self.head
        for i in range(self.length()):
            if temp.data==data:
                break
            else:
                temp=temp.next
                
def prime(size):
    mini=None
    for i in range(size,2*size):
        for j in range(2,i+1):
            if i%j==0 and i!=j:
                break
            elif i%j!=0 and i!=j:
                continue
            else:
                mini=i
        if mini==i:
            break
    return mini

# Hash table definition
class HashTable:
    def __init__(self,size):
        self.size=size
        self.array=np.array([None for i in range(size)])
        self.length=0
        self.r1=random.randint(1,2**30)%self.size
        self.r2=random.randint(1,2**30)%self.size
        self.r3=random.randint(1,2**30)%self.size
        self.r4=random.randint(1,2**30)%self.size
        
    def binary(self,x):
        int_to_four_bytes=struct.Struct('<I').pack
        x1,x2,x3,x4=int_to_four_bytes(x & 0xFFFFFFFF)
        return np.array([x1,x2,x3,x4])
    
    def hashf(self,data):
        binary_arr=self.binary(data)
        
        rand=np.array([self.r1,self.r2,self.r3,self.r4])
        index=sum(rand*binary_arr)%self.size
        return index
        
    def insert(self,data):
        if not self.array[self.hashf(data)]:
            self.array[self.hashf(data)]=LinkedList()
        self.array[self.hashf(data)].insert_data(data)
    
    def search(self,data):
        if self.array[self.hashf(data)]:
            self.array[self.hashf(data)].search_data(data)