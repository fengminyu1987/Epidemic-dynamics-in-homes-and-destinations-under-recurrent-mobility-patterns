import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 显示中文标签
plt.rcParams['axes.unicode_minus']=False   # 解决保存图像是负号'-'显示为方块的问题

class SIRMonteCarlo:
    def __init__(self, N, beta, gama, infect_distance, movement_distance, num_initial_infected, fps):
        self.N = N
        self.beta = beta
        self.gama = gama
        self.infect_distance = infect_distance
        self.movement_distance = movement_distance
        self.num_initial_infected = num_initial_infected
        self.fps = fps

        # 初始化位置
        self.positions = np.random.rand(N, 2) * np.array([800, 500])  # 位置范围为[0, 800] x [0, 500]
        self.initial_infected = np.random.choice(N, num_initial_infected, replace=False)
        self.status = np.zeros(N)  # 0表示易感，1表示感染，2表示恢复
        self.status[self.initial_infected] = 1

        # 创建动画
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 800)
        self.ax.set_ylim(0, 500)
        self.sc = self.ax.scatter(self.positions[:, 0], self.positions[:, 1], c=self.status, cmap='viridis', s=50)
        self.txtN = self.ax.text(10, 480, f'总人口: {N}', color='red', fontsize=12, fontweight='bold')
        self.txtI = self.ax.text(10, 480-25, f'携带者: {num_initial_infected}', color='red', fontsize=12, fontweight='bold')
        self.txtR = self.ax.text(10, 480-50, '隔离或具抗性者: 0', color='red', fontsize=12, fontweight='bold')

        self.ani = FuncAnimation(self.fig, self.update, frames=np.arange(1, 200), interval=1000/fps)

    def update(self, frame):
        theta = np.random.rand(self.N) * 2 * np.pi * self.movement_distance
        move_direction = np.array([np.cos(theta), np.sin(theta)]).T
        self.positions += move_direction

        # 感染过程
        for i in range(self.N):
            if self.status[i] == 1:
                for j in range(self.N):
                    if self.status[j] == 0 and np.linalg.norm(self.positions[i] - self.positions[j]) <= self.infect_distance:
                        if np.random.rand() < self.beta:
                            self.status[j] = 1

        # 恢复过程
        for i in range(self.N):
            if self.status[i] == 1 and np.random.rand() < self.gama:
                self.status[i] = 2

        self.sc.set_offsets(self.positions)
        self.sc.set_array(self.status)
        self.txtI.set_text(f'携带者: {np.sum(self.status == 1)}')
        self.txtR.set_text(f'隔离或具抗性者: {np.sum(self.status == 2)}')

        return self.sc, self.txtI, self.txtR

    def show(self):
        plt.show()

# 参数设置
N = 500  # 总人口
beta = 0.8  # 感染率
gama = 0.01  # 恢复率
infect_distance = 10  # 感染距离
movement_distance = 10  # 移动距离
num_initial_infected = 50  # 初始感染人数
fps = 10  # 动画帧率

# 运行模拟
simulator = SIRMonteCarlo(N, beta, gama, infect_distance, movement_distance, num_initial_infected, fps)
simulator.show()
