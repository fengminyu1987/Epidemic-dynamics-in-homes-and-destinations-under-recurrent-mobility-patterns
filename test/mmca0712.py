import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# 设置随机种子以确保结果可复现
np.random.seed(42)
random.seed(42)

# 生成小世界网络
N = 50
k = 4
p_rewire = 0.1
G = nx.watts_strogatz_graph(N, k, p_rewire)

# 将30个节点设为Residence，20个节点设为Destination
residence_count = 30
destination_count = 20

# 随机分配节点类型
node_types = np.random.choice(['Residence', 'Destination'], size=N, p=[0.6, 0.4])

# 为节点添加类型属性
for i in range(N):
    G.nodes[i]['type'] = node_types[i]

# 生成每个子种群的网络并计算平均度
residence_size = 100
destination_size = 50
subpopulations = {}
average_degrees = np.zeros(N)

for i in range(N):
    if G.nodes[i]['type'] == 'Residence':
        subgraph = nx.erdos_renyi_graph(residence_size, 0.1)
    else:
        subgraph = nx.complete_graph(destination_size)
    subpopulations[i] = subgraph
    average_degrees[i] = np.mean([d for n, d in subgraph.degree()])

# 初始化感染状态
initial_infected_density = 0.01
rho = np.zeros(N)
for i in range(N):
   rho[i] = initial_infected_density 
    # if G.nodes[i]['type'] == 'Residence':
    #     rho[i] = initial_infected_density

# 定义传染率和恢复率
beta = 0.1
mu = 0.1
p_move = 0.1

# 定义邻接矩阵
W = nx.adjacency_matrix(G).toarray()

# 计算C矩阵
def calculate_C_matrix(W):
    N = W.shape[0]
    C = np.zeros_like(W, dtype=float)
    for i in range(N):
        row_sum = np.sum(W[i, :])
        if row_sum > 0:
            C[i, :] = W[i, :] / row_sum
    return C

C = calculate_C_matrix(W)

# 计算n_k_to_j矩阵
def get_n_k_to_j(n, p, C):
    N = len(n)
    n_k_to_j = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            if k == j:
                n_k_to_j[k, j] = (1 - p) * n[j]
            else:
                n_k_to_j[k, j] = p * C[k, j] * n[k]
    return n_k_to_j

# 设置子种群大小
n = np.zeros(N)
for i in range(N):
    if G.nodes[i]['type'] == 'Residence':
        n[i] = residence_size
    else:
        n[i] = destination_size

# 初始化感染历史记录
steps = 100
rho_history = np.zeros((steps, N))

# 初始化居住点i的感染率R_i和目的地感染率D_i的合集P_i
# R_i = np.zeros(residence_count)
# D_i = np.zeros(destination_count)
P_i = np.zeros(N)

# 更新居住点感染率 +更新目的地感染率
def update_P_i(rho, p_move, n_k_to_j, C):
    for i in range(N):
        if G.nodes[i]['type'] == 'Residence':
            P_i[i] = 1 - (1 - beta) ** ((1 - p_move) * average_degrees[i] * rho[i] +
                                         np.sum([p_move * C[j, i] * average_degrees[j] * rho[j] for j in range(N)]))
        else: 
            # P_i[i] = 1 - (1 - beta) ** (np.sum([n_k_to_j[k, i] * rho[k] for k in range(N)]))
            P_i[i] = 1 - (1 - beta) **((1 - p_move) * n[i] * rho[i] +
                                         np.sum([p_move * C[j, i] * n[j] * rho[j] for j in range(N)]))

    return P_i


for t in range(steps):
    rho_history[t, :] = rho
    new_rho = np.zeros(N)
    n_k_to_j = get_n_k_to_j(n, p_move, C)
    P_i = update_P_i(rho, p_move, n_k_to_j, C)
    
    for i in range(N):
        if G.nodes[i]['type'] == 'Residence':
            Pi_effective = (1 - p_move) * P_i[i] + p_move * np.sum([C[i, j] * P_i[j] for j in range(N)])

            new_rho[i] = (1 - mu) * rho[i] + (1 - rho[i]) * Pi_effective
    
    rho = new_rho

# 计算平均感染比例
I_avg = np.mean(rho_history, axis=1)
S_avg = 1 - I_avg

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(range(steps), I_avg, label="Infected (I)", color='red')
plt.plot(range(steps), S_avg, label="Susceptible (S)", color='blue')
plt.xlabel("Time Steps")
plt.ylabel("Average Population Proportion")
plt.title("SIS Metapopulation Model with Residence and Destination Nodes")
plt.legend()
plt.show()
