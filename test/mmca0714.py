# 这个类编写的有点问题
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# 设置随机种子以确保结果可复现
np.random.seed(42)
random.seed(42)

# 生成小世界网络
N = 20
k = 8
p_rewire = 0.8
G = nx.watts_strogatz_graph(N, k, p_rewire)

# 为每条边分配1到10之间的随机权重
for u, v in G.edges():
    G[u][v]["weight"] = random.randint(1, 10)

# 获取邻接矩阵并包含权重
W = nx.to_numpy_array(G, weight="weight")


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

# 生成每个子种群的网络并计算平均度
n = np.random.randint(50, 101, size=N)  # 每个节点的初始人口
average_degrees = np.zeros(N)

for i in range(N):
    average_degrees[i] = n[i] * 0.1

# 初始化感染状态
initial_infected_density = 0.01
rho = np.zeros(N)
for i in range(N):
    rho[i] = initial_infected_density

# 定义传染率和恢复率
beta = 0.0025
mu = 0.1
p_move = 0.1
alpha = 0.6

# 初始化感染历史记录
steps = 100
rho_history = np.zeros((steps, N))

for t in range(steps):
    rho_history[t, :] = rho
    new_rho = np.zeros(N)

    for i in range(N):
        R_i = 1 - (1 - beta) ** (
            (1 - p_move) * average_degrees[i] * rho[i]
            + np.sum([p_move * C[j, i] * average_degrees[j] * rho[j] for j in range(N)])
        )
        D_i = 1 - (1 - beta) ** (
            (1 - p_move) * n[i] * rho[i]
            + np.sum([p_move * C[j, i] * n[j] * rho[j] for j in range(N)])
        )
        P_i = alpha * R_i + (1 - alpha) * D_i
        Pi_i = (1 - p_move) * P_i + p_move * np.sum([C[i, j] * P_i for j in range(N)])
        new_rho[i] = (1 - mu) * rho[i] + (1 - rho[i]) * Pi_i

    rho = new_rho

# 计算平均感染比例
I_avg = np.mean(rho_history, axis=1)
S_avg = 1 - I_avg
print(I_avg[steps - 1])

# 绘制结果
# plt.figure(figsize=(12, 6))
# plt.plot(range(steps), I_avg, label="Infected (I)", color='red')
# plt.plot(range(steps), S_avg, label="Susceptible (S)", color='blue')
# plt.xlabel("Time Steps")
# plt.ylabel("Average Population Proportion")
# plt.title("SIS Metapopulation Model with Residence and Destination Nodes")
# plt.legend()
# plt.show()
