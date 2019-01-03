#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
import pandas as pd
import time


# In[2]:


##Import data
G=nx.read_gml('adjnoun.gml')


# In[3]:


##Encode node strings in a dictionary
node=list(G.node)
n=len(node)
node_dict={}
node_dictr={}

for i in range(n):
    node_dict[node[i]]=i
    node_dictr[i]=node[i]


# In[ ]:





# In[4]:


##Adjacency matrix
adj_mat=np.zeros((n,n))
edg=list(G.edges)
for i in edg:
    j=node_dict[i[0]]
    k=node_dict[i[1]]
    adj_mat[j,k]=1
    adj_mat[k,j]=1


# In[5]:


adj_mat


# In[6]:


##Degree computation
deg=np.sum(adj_mat,axis=0)
if(np.any(deg<0)==False):
    print("Graph is connected")
else:
    print("Graph is not connected")


# In[ ]:





# In[7]:


##Median degree
print("Median degree: "+str(np.median(deg)))


# In[8]:


##Numer of triangles
count=0
for i in range(n):
    for j in range(n):
        for k in range(n):
            if(i!=j and j!=k and i!=k and adj_mat[i,j] and adj_mat[j,k] and adj_mat[k,i]):
                count += 1
print("Number of triangles: "+str(count/6))


# In[9]:


cost_graph=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if (adj_mat[i,j]==1):
            cost_graph[i,j]=abs(i-j)
cost_graph


# # Kruskal's Algorithm

# In[10]:


##Upper triangular matrix form of cost graph
def upper_triangle(n,cost_graph):
    up_cost=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            up_cost[i,j]=cost_graph[i,j]
    return up_cost


# In[11]:


##Initial tree set (n possible trees set with each set containing one node)
def initial_set(n1):
    lst=[]
    for i in range(n1):
        lst.append([i])
    return lst


# In[12]:


##Find all the non-zero elements and appending them into a list by a tuple containing node i, node j, cost of ij)
def index_value_append(up_cst):
    non_zer=np.nonzero(up_cst)
    val=up_cost[non_zer]

    indx_val=[]
    for i in range(len(non_zer[0])):
        indx_val.append((non_zer[0][i],non_zer[1][i],val[i]))
    return indx_val


# In[13]:


##Sort the tuple list based on the cost of ij 
def sort_value(indx_val):
    srt=sorted(indx_val, key= lambda t:t[2])
    return srt


# In[14]:


##Find to which tree set the nodes i and j belong
def find_set(u1,v1,N1):
    for i in range(len(N1)):
        for j in range(len(N1[i])):
            if (u1==N1[i][j]):
                set_u=i
            if (v1==N1[i][j]):
                set_v=i
    return set_u, set_v


# In[15]:


##Merge tree sets containing nodes i and j into a single list 
def N_append(set_u1,set_v1,N1):
    for i in range(len(N1[set_v1])):
        N1[set_u1].append(N1[set_v1][i])
    N1.remove(N[set_v1])
    return N1


# In[16]:


##ALGORITHM
time1=time.time()
assign=[]
tree_length=[]
up_cost=upper_triangle(n,cost_graph)
N=initial_set(n)
ind_val=index_value_append(up_cost)
sorted_value=sort_value(ind_val)
n_count=0  

while (n_count<(n-1)):
    u=sorted_value[0][0]
    v=sorted_value[0][1]
    set_u,set_v=find_set(u,v,N)
    if (set_u==set_v):
        sorted_value.remove(sorted_value[0])
    else:
        assign.append([u,v])
        N=N_append(set_u,set_v,N)
        tree_length.append(sorted_value[0][2])
        sorted_value.remove(sorted_value[0])
        n_count+=1
        
time2=time.time()
time_kruskal=time2-time1


# In[17]:


##Assign the node names back to the numbered nodes 
assign_name=[]
for i in assign:
    assign_name.append([node_dictr[i[0]],node_dictr[i[1]]])


# In[18]:


print("Minimal spanning tree contains the following arcs (Optimal path): \n")
print(assign_name)


# In[19]:


print("Length of the minimal spanning tree (Optimal cost) is : "+str(sum(tree_length)))


# In[20]:


print("Running Time is :"+str(time_kruskal))


# In[ ]:





# # Prim's Algorithm

# In[21]:


def prep_matrix(n,cost_grph):
    #mat1=upper_triangle(n,cost_graph)
    for l in range(n):
        for l1 in range(n):
            if(cost_grph[l,l1]==0):
                cost_grph[l,l1]=np.max(cost_grph)+10
    return cost_grph


# In[22]:


def select_initial_node(n1):
    return np.random.randint(0,n1)


# In[23]:


time3=time.time()
cost_mat=prep_matrix(n,np.array(cost_graph))
init_node=select_initial_node(n)
#init_node=0
selected_node=[init_node]
path=[]
length=[]
temp_mat=np.reshape(cost_mat[init_node,:],(1,-1))
m=np.max(cost_graph)+10
temp_mat[:,selected_node[0]]=m
cost_mat[:,selected_node[0]]=m

while (len(selected_node)<n):
    minimum=np.amin(temp_mat,axis=1)
    arg_minimum=np.argmin(temp_mat,axis=1)
    all_min=np.amin(minimum)
    all_argmin=np.argmin(minimum)
    selected_node.append(arg_minimum[all_argmin])
    path.append([selected_node[all_argmin],arg_minimum[all_argmin]])
    length.append(all_min)
    
    temp_mat[:,arg_minimum[all_argmin]]=m
    cost_mat[:,arg_minimum[all_argmin]]=m
    temp_mat=np.vstack((temp_mat,np.reshape(cost_mat[arg_minimum[all_argmin],:],(1,-1))))
time4=time.time()
time_prim=time4-time3


# In[24]:


##Assign the node names back to the numbered nodes 
path_name=[]
for i in path:
    path_name.append([node_dictr[i[0]],node_dictr[i[1]]])


# In[25]:


print("Minimal spanning tree contains the following arcs (Optimal path): \n")
print(path_name)


# In[26]:


print("Length of the minimal spanning tree (Optimal cost) is : "+str(sum(length)))


# In[27]:


print("Running Time is :"+str(time_prim))


# In[28]:


table={'Method':['Kruskal Algorithm','Prims Algorithm'],'Optimal Cost':[sum(tree_length),sum(length)],'Running Time':[time_kruskal,time_prim]}
#table=[{'Method':'Kruskal Algorithm','Optimal Cost':sum(tree_length)},{'Method':'Prims Algorithm','Optimal Cost':sum(length)}]
pd.DataFrame(table)

