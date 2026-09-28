import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 参数设置
N = 10  # 节点数量
P = 1000  # 总人口
residence_ratio = 0.4
num_residence = int(N * residence_ratio)
num_destination = N - num_residence

# 居住点和目的地节点划分
residence_nodes = np.random.choice(N, num_residence, replace=False)
destination_nodes = np.array([i for i in range(N) if i not in residence_nodes])

# 居住点和目的地的个体数量
n = np.random.randint(50, 150, N)
k_residence = 10  # 平均度
k_destination = N - 1  # 平均度为完全图

# 构建WS小世界网络
G = nx.watts_strogatz_graph(N, k_residence, 0.1)
W = nx.to_numpy_array(G)

# 邻接矩阵归一化
W = W / W.sum(axis=1, keepdims=True)


# 仿真参数
beta = 0.003  # 感染率
mu = 0.1  # 恢复率
p = 0.2  # 移动概率
steps = 100  # 仿真步数

# 初始化感染率
rho = np.random.rand(N)
rho_history = np.zeros((steps, N))

# 模拟
for t in range(steps):
    # 记录当前感染率
    rho_history[t] = rho

    # 计算感染概率
    P_i = np.zeros(N)
    for i in range(N):
        if i in residence_nodes:
            # 居住点
            product = 1.0
            for j in range(N):
                if i == j:
                    n_j_to_i = (1 - p) * n[i]
                else:
                    n_j_to_i = p * W[j, i] * n[j]
                product *= (1 - beta * rho[j]) ** n_j_to_i
            R_i = 1 - (1 - beta) ** (
                (1 - p) * k_residence * rho[i]
                + sum(p * W[j, i] * k_residence * rho[j] for j in range(N))
            )
            P_i[i] = (1 - p) * R_i + p * sum(
                W[i, j] * (1 - (1 - beta) ** sum(n[k] * rho[k] for k in range(N)))
                for j in range(N)
            )
        else:
            # 目的地
            D_j = 1 - (1 - beta) ** sum(n[k] * rho[k] for k in range(N))
            P_i[i] = (1 - p) * D_j + p * sum(
                W[i, j] * (1 - (1 - beta) ** sum(n[k] * rho[k] for k in range(N)))
                for j in range(N)
            )

    # 更新感染率
    new_rho = np.zeros(N)
    for i in range(N):
        new_rho[i] = (1 - mu) * rho[i] + (1 - rho[i]) * P_i[i]
    rho = new_rho

# 计算平均感染和易感比例
I_mean = rho_history.mean(axis=1)
S_mean = 1 - I_mean

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(range(steps), I_mean, label="Infected (I)", color="red")
plt.plot(range(steps), S_mean, label="Susceptible (S)", color="green")
plt.xlabel("Time")
plt.ylabel("Average Proportion")
plt.title("SIS Metapopulation Model: Evolution of S and I Proportions")
plt.legend()
plt.show()
