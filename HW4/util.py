import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
from numpy import Inf

# Linked List definition
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

# 城市節點定義  
class vertex:
    def __init__(self,x,y):
        self.id=0
        self.x=x
        self.y=y
        self.neighbors=[]
    
    def add_neighbor(self,nv):
        self.neighbors.append([nv.id,((nv.x-self.x)**2+(nv.y-self.y)**2)**(0.5)])
        nv.neighbors.append([self.id,((nv.x-self.x)**2+(nv.y-self.y)**2)**(0.5)])

    def print_neighbors(self):
        return self.neighbors
            
    def __str__(self):
        return f"({self.x},{self.y})"
    
    def __repr__(self):
        return f"{self.id}"

# Graph definition        
class graph:
    def __init__(self,n):
        self.n=n
        self.vertices,self.edges,self.v_mega,self.v_second,self.v_rural=[[] for i in range(5)]
        self.AdjacencyList={}
        
        #主要城市和主要城市之間一定會有道路
        self.megacity=[]
        for i in range(int(len(self.v_mega)/2)):
            self.megacity.append(self.v_mega[i*2+1][0])
        if len(self.megacity)>1:
            x=itertools.combinations(self.megacity,2)
            for i,j in list(x):
                self.add_edge(i,j)

        #主要城市的區連到主要城市
        for i in range(int(len(self.v_mega)/2)):
            for j in self.v_mega[i*2+1]:
                self.add_edge(self.v_mega[i*2],j)
                
        #次要城市連到次要城市
        self.secondarycity=[]
        for i in range(int(len(self.v_second)/2)):
            self.secondarycity.append(self.v_second[i*2])
        if len(self.secondarycity)>1:
            x=itertools.combinations(self.secondarycity,2)
            for i,j in list(x):
                self.add_edge(i,j)
                
        #次要城市的區連到主要城市
        for i in range(int(len(self.v_second)/2)):
            for j in self.v_second[i*2+1]:
                self.add_edge(self.v_second[i*2],j)


    def add_vertex(self,new):
        if new not in self.vertices:
            self.vertices.append(new)
        else:
            print("Vertex already exists in the graph")
            
    def generate_vertices(self):
        # Megacity
        for i in range(int(self.n*0.02+np.random.rand()*3)):
            new_v=vertex(random.randint(-90,91),random.randint(-90,91))
            self.add_vertex(new_v)
            self.v_mega.append(new_v)
            self.v_mega.append([])
            for j in range(int(self.n*0.09)):
                temp_v=vertex(int((new_v.x-20)+np.random.rand()*40),int((new_v.y-20)+np.random.rand()*40))
                self.add_vertex(temp_v)
                self.v_mega[1+2*i].append(temp_v)
        
        #Secondary city
        for i in range(int(self.n*0.06+np.random.rand()*3)):
            new_v=vertex(random.randint(-90,91),random.randint(-90,91))
            self.add_vertex(new_v)
            self.v_second.append(new_v)
            self.v_second.append([])
            for j in range(int(self.n*0.04)):
                temp_v=vertex(int((new_v.x-10)+np.random.rand()*20),int((new_v.y-10)+np.random.rand()*20))
                self.add_vertex(temp_v)
                self.v_second[1+2*i].append(temp_v)
                
        #rural area
        for i in range(self.n-len(self.vertices)):
            new_v=vertex(random.randint(-100,101),random.randint(-100,101))
            self.add_vertex(new_v)
            self.v_rural.append(new_v)
        
    def print_vertices(self,mode=None):
        if mode=="mega":
            return self.v_mega
        elif mode=="second":
            return self.v_second
        elif mode=="rural":
            return self.v_rural
        else:
            return self.vertices
    
    def add_edge(self,v1,v2):
        self.edges.append([v1,v2])
        v1.add_neighbor(v2)
    

    def generate_edges(self):
        
        #主要城市的區連到主要城市
        for i in range(int(len(self.v_mega)/2)):
            for j in self.v_mega[i*2+1]:
                self.add_edge(self.v_mega[i*2],j)
                
        #主要城市和主要城市之間一定會有道路
        self.megacity=[]
        for i in range(int(len(self.v_mega)/2)):
            self.megacity.append(self.v_mega[i*2])
        if len(self.megacity)>1:
            x=itertools.combinations(self.megacity,2)
            for i,j in list(x):
                self.add_edge(i,j)         
                
        #隨機
        x=itertools.combinations(self.print_vertices(),2)
        for i,j in list(x):
            if len(list(i.print_neighbors().keys()))<self.n*0.05 and len(list(i.print_neighbors().keys()))<self.n*0.05:
                if [i,j] or [j,i] not in self.print_edges():
                    if ((i.x-j.x)**2+(i.y-j.y)**2)**0.5<(30+np.random.rand()*30):
                        self.add_edge(i,j)
                        
        for i in self.print_vertices():
            self.AdjacencyList[i]=dict(i.print_neighbors())
                
    def print_edges(self):
        return self.edges
    
    def print_graph(self):
        x,y=[[] for i in range(2)]
        for i in self.print_vertices():
            x.append(i.x)
            y.append(i.y)
        plt.figure(figsize=(10,10))
        plt.scatter(x,y)
        
        for i in range(len(self.print_edges())):
            plt.plot([self.print_edges()[i][0].x,self.print_edges()[i][1].x],
                     [self.print_edges()[i][0].y,self.print_edges()[i][1].y],color="b")
            
    def check(self):
        a=self.print_edges()
        d=DisjointSet()
        for e in a:
            d.add(*e)
        pivot=list(d.leader.values())[0]
        count=0
        for i in list(d.leader.values()):
            if i==pivot:
                count+=1
        for i in self.print_vertices():
            if len(i.print_neighbors())==0:
                count-=1
        if count!=len(d.leader.values()):
            return "需重新產生網路圖"
        elif count==len(d.leader.values()):
            return "成功"
    
    def __repr__(self):
        return str(self.AdjacencyList)
    
# Disjoint Set Definition
class DisjointSet(object):
    
    def __init__(self,size=None):
        if size is None:
            self.leader = {}  # maps a member to the group's leader
            self.group = {}  # maps a group leader to the group (which is a set)
            self.oldgroup = {}
            self.oldleader = {}
        else:
            self.group = { i:set([i]) for i in range(0,size) }
            self.leader = { i:i for i in range(0,size) }
            self.oldgroup = { i:set([i]) for i in range(0,size) }
            self.oldleader = { i:i for i in range(0,size) }                

    def add(self, a, b):
        self.oldgroup = self.group.copy()
        self.oldleader = self.leader.copy()
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb:
                    return  # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])

    def connected(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                return leadera == leaderb
            else:
                return False
        else:
            return False

    def undo(self):        
        self.group = self.oldgroup.copy()
        self.leader = self.oldleader.copy()
        
# generating function
def generate_graph(n):
    g=graph(n)
    g.generate_vertices()
    g.generate_edges()
    output=g.check()
    if output=="需重新產生網路圖":
        return generate_graph(n)
    elif output=="成功":
        return g
    
def show(n):
    g=generate_graph(n)
    print("節點數量：",g.n)
    print("主要城市數量：",int(len(g.v_mega)/2))
    print("次要城市數量：",int(len(g.v_second)/2))
    print("鄉村數量：",int(len(g.v_rural)))
    g.print_graph()
    
# Heap definition
class Heap():
 
    def __init__(self):
        self.array = []
        self.size = 0
        self.pos = []
 
    def newMinHeapNode(self, v, dist):
        minHeapNode = [v, dist]
        return minHeapNode
 
    # A utility function to swap two nodes
    # of min heap. Needed for min heapify
    def swapMinHeapNode(self, a, b):
        t = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = t
        
    def minHeapify(self, idx):
        smallest = idx
        left = 2*idx + 1
        right = 2*idx + 2
 
        if (left < self.size and
           self.array[left][1]
            < self.array[smallest][1]):
            smallest = left
 
        if (right < self.size and
           self.array[right][1]
            < self.array[smallest][1]):
            smallest = right
 
        # The nodes to be swapped in min
        # heap if idx is not smallest
        if smallest != idx:
 
            # Swap positions
            self.pos[self.array[smallest][0]] = idx
            self.pos[self.array[idx][0]] = smallest
 
            # Swap nodes
            self.swapMinHeapNode(smallest, idx)
            self.minHeapify(smallest)
            
    def extractMin(self):
 
        # Return NULL wif heap is empty
        if self.isEmpty() == True:
            return
 
        # Store the root node
        root = self.array[0]
 
        # Replace root node with last node
        lastNode = self.array[self.size - 1]
        self.array[0] = lastNode
 
        # Update position of last node
        self.pos[lastNode[0]] = 0
        self.pos[root[0]] = self.size - 1
 
        # Reduce heap size and heapify root
        self.size -= 1
        self.minHeapify(0)
 
        return root

    def isEmpty(self):
        return True if self.size == 0 else False
 
    def decreaseKey(self, v, dist):
 
        # Get the index of v in  heap array
 
        i = self.pos[v]
 
        # Get the node and update its dist value
        self.array[i][1] = dist
 
        # Travel up while the complete tree is
        # not hepified. This is a O(Logn) loop
        while (i > 0 and self.array[i][1] <
                  self.array[(i - 1) // 2][1]):
 
            # Swap this node with its parent
            self.pos[ self.array[i][0] ] = (i-1)//2
            self.pos[ self.array[(i-1)//2][0] ] = i
            self.swapMinHeapNode(i, (i - 1)//2 )
 
            # move to parent index
            i = (i - 1) // 2;
        
# A utility function to check if a given
    # vertex 'v' is in min heap or not
    def isInMinHeap(self, v):
 
        if self.pos[v] < self.size:
            return True
        return False
 
# print array definition
def printArr(dist, n):
    print ("Vertex\tDistance from source")
    for i in range(n):
        print ("%d\t\t%d" % (i,dist[i]))
       
# factorial definition 
def factorial(n):
    if n==1 or n==0:
        return n
    else:
        return n*factorial(n-1)
    
# average definition
def average(n):
    a=generate_graph(n)
    total=0
    for i in list(a.AdjacencyList.keys()):
        s=a.dijkstra(i)
        total+=s
    average=total/(n*(n-1)/2)
    #print("平均距離：",average)
    return average

# dijkstras definition
def naive_dijkstras(graph,root):
    n = len(graph)
    # initialize distance list as all infinities
    dist = [Inf for _ in range(n)]
    # set the distance for the root to be 0
    dist[root] = 0
    # initialize list of visited nodes
    visited = [False for _ in range(n)]
    # loop through all the nodes
    for _ in range(n):
        # "start" our node as -1 (so we don't have a start node yet)
        u = -1
        # loop through all the nodes to check for visitation status
        for i in range(n):
            # if the node 'i' hasn't been visited and
            # we haven't processed it or the distance we have for it is less
            # than the distance we have to the "start" node
            if not visited[i] and (u == -1 or dist[i] < dist[u]):
                u = i
        # all the nodes have been visited or we can't reach this node
        if dist[u] == Inf:
            break
        # set the node as visited
        visited[u] = True
        # compare the distance to each node from the "start" node
        # to the distance we currently have on file for it
        for v, l in graph[u]:
            if dist[u] + l < dist[v]:
                dist[v] = dist[u] + l
    return dist

# Fibonacci Heap definition
class FibonacciHeap:
    # internal node class 
    class Node:
        def __init__(self, data):
            self.data = data
            self.parent = self.child = self.left = self.right = None
            self.degree = 0
            self.mark = False
 
    # function to iterate through a doubly linked list
    def iterate(self, head):
        node = stop = head
        flag = False
        while True:
            if node == stop and flag is True:
                break
            elif node == stop:
                flag = True
            yield node
            node = node.right
 
    # pointer to the head and minimum node in the root list
    root_list, min_node = None, None
 
    # maintain total node count in full fibonacci heap
    total_nodes = 0
 
    # return min node in O(1) time
    def find_min(self):
        return self.min_node
 
    # extract (delete) the min node from the heap in O(log n) time
    # amortized cost analysis can be found here (http://bit.ly/1ow1Clm)
    def extract_min(self):
        z = self.min_node
        if z is not None:
            if z.child is not None:
                # attach child nodes to root list
                children = [x for x in self.iterate(z.child)]
                for i in range(0, len(children)):
                    self.merge_with_root_list(children[i])
                    children[i].parent = None
            self.remove_from_root_list(z)
            # set new min node in heap
            if z == z.right:
                self.min_node = self.root_list = None
            else:
                self.min_node = z.right
                self.consolidate()
            self.total_nodes -= 1
        return z
 
    # insert new node into the unordered root list in O(1) time
    def insert(self, data):
        n = self.Node(data)
        n.left = n.right = n
        self.merge_with_root_list(n)
        if self.min_node is None or n.data < self.min_node.data:
            self.min_node = n
        self.total_nodes += 1
 
    # modify the data of some node in the heap in O(1) time
    def decrease_key(self, x, k):
        if k > x.data:
            return None
        x.data = k
        y = x.parent
        if y is not None and x.data < y.data:
            self.cut(x, y)
            self.cascading_cut(y)
        if x.data < self.min_node.data:
            self.min_node = x
 
    # merge two fibonacci heaps in O(1) time by concatenating the root lists
    # the root of the new root list becomes equal to the first list and the second
    # list is simply appended to the end (then the proper min node is determined)
    def merge(self, h2):
        H = FibonacciHeap()
        H.root_list, H.min_node = self.root_list, self.min_node
        # fix pointers when merging the two heaps
        last = h2.root_list.left
        h2.root_list.left = H.root_list.left
        H.root_list.left.right = h2.root_list
        H.root_list.left = last
        H.root_list.left.right = H.root_list
        # update min node if needed
        if h2.min_node.data < H.min_node.data:
            H.min_node = h2.min_node
        # update total nodes
        H.total_nodes = self.total_nodes + h2.total_nodes
        return H
 
    # if a child node becomes smaller than its parent node we
    # cut this child node off and bring it up to the root list
    def cut(self, x, y):
        self.remove_from_child_list(y, x)
        y.degree -= 1
        self.merge_with_root_list(x)
        x.parent = None
        x.mark = False
 
    # cascading cut of parent node to obtain good time bounds
    def cascading_cut(self, y):
        z = y.parent
        if z is not None:
            if y.mark is False:
                y.mark = True
            else:
                self.cut(y, z)
                self.cascading_cut(z)
 
    # combine root nodes of equal degree to consolidate the heap
    # by creating a list of unordered binomial trees
    def consolidate(self):
        A = [None] * self.total_nodes
        nodes = [w for w in self.iterate(self.root_list)]
        for w in range(0, len(nodes)):
            x = nodes[w]
            d = x.degree
            while A[d] != None:
                y = A[d] 
                if x.data > y.data:
                    temp = x
                    x, y = y, temp
                self.heap_link(y, x)
                A[d] = None
                d += 1
            A[d] = x
        # find new min node - no need to reconstruct new root list below
        # because root list was iteratively changing as we were moving 
        # nodes around in the above loop
        for i in range(0, len(A)):
            if A[i] is not None:
                if A[i].data < self.min_node.data:
                    self.min_node = A[i]
 
    # actual linking of one node to another in the root list
    # while also updating the child linked list
    def heap_link(self, y, x):
        self.remove_from_root_list(y)
        y.left = y.right = y
        self.merge_with_child_list(x, y)
        x.degree += 1
        y.parent = x
        y.mark = False
 
    # merge a node with the doubly linked root list   
    def merge_with_root_list(self, node):
        if self.root_list is None:
            self.root_list = node
        else:
            node.right = self.root_list.right
            node.left = self.root_list
            self.root_list.right.left = node
            self.root_list.right = node
 
    # merge a node with the doubly linked child list of a root node
    def merge_with_child_list(self, parent, node):
        if parent.child is None:
            parent.child = node
        else:
            node.right = parent.child.right
            node.left = parent.child
            parent.child.right.left = node
            parent.child.right = node
 
    # remove a node from the doubly linked root list
    def remove_from_root_list(self, node):
        if node == self.root_list:
            self.root_list = node.right
        node.left.right = node.right
        node.right.left = node.left
 
    # remove a node from the doubly linked child list
    def remove_from_child_list(self, parent, node):
        if parent.child == parent.child.right:
            parent.child = None
        elif parent.child == node:
            parent.child = node.right
            node.right.parent = parent
        node.left.right = node.right
        node.right.left = node.left
        
# Fibonacci Heap Dijkstra definition
def fdijkstra(adjList, source, sink = None):
    n = len(adjList)    #intentionally 1 more than the number of vertices, keep the 0th entry free for convenience
    visited = [False]*n
    distance = [float('inf')]*n

    heapNodes = [None]*n
    heap = FibonacciHeap()
    for i in range(1, n):
        heapNodes[i] = heap.insert(float('inf'), i)     # distance, label

    distance[source] = 0
    heap.decrease_key(heapNodes[source], 0)

    while heap.total_nodes:
        current = heap.extract_min().value
        visited[current] = True

        #early exit
        if sink and current == sink:
            break

        for (neighbor, cost) in adjList[current]:
            if not visited[neighbor]:
                if distance[current] + cost < distance[neighbor]:
                    distance[neighbor] = distance[current] + cost
                    heap.decrease_key(heapNodes[neighbor], distance[neighbor])


    return distance