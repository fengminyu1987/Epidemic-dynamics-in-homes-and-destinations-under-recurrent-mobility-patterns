# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import math

class mmcaMeta_adjust:
    def __init__(self, W , p , epi_params, n_vec , patch_ids):
        # W = np.random.rand(2,2)
        ## 表示模型中的区域数量，即人口数量数组的长度 N=2
        self.N = len(n_vec)

        ## 表示拟合期的长度，即移动性矩阵列表的长度 tau=2
        self.tau = len(W)

        ## 检查输入
        # assert len(W) == self.tau, "The temporal extent of mobility matrices must match that of average contacts."
        assert len(W[0]) == self.N, "The dimensions of the mobility matrices do not match that of the population size array."

        ## 初始化 Wij表示社区间的边的权重
        self.W = W 
        #一个包含1 x N数组的数组,数组的每个元素表示给定区域内的人口数量
        self.n_vec = n_vec
        #每个区域的patch id列表,在多个patch中的每个patch都有一个patch id
        self.patch_ids = patch_ids
        # 移动率
        self.p = p

        self.Lambda = epi_params['Lambda']
        self.beta = epi_params['beta']
        self.mu = epi_params['mu']

    ## 计算节点在社区i与j之间的流动率Cij 移动矩阵C 
    def count_C(self):

        N = self.N
        W = self.W

        # 计算W每行的和row_sums
        row_sums = np.sum(W, axis=1)
    
        # 计算C矩阵 Cij = Wij / Σ(l=1->N) Wil
        C =  W / row_sums[:, np.newaxis] 

        self.C = C
        # print("当前C矩阵为:", C)
        return C


    #设置了每个舱室中存在的初始比例，以便启动马尔可夫过程
    def set_seed(self, E_0, I_0, startTime):

        #初始化
        N = self.N
        n_vec = self.n_vec
        patch_ids = self.patch_ids


        ## Initialize numpy arrays with zeros
        rho_S = np.zeros(N)
        rho_E = np.zeros(N)
        rho_I = np.zeros(N)
        rho_R = np.zeros(N)

        ## 根据infectious seeds更新 S, E, 和 Ia Ic 舱室
        for i in range(N):
            rho_E[i] = E_0[i] / n_vec[i]
            rho_I[i] = I_0[i] / n_vec[i]
            rho_S[i] = 1 - (rho_E[i] + rho_I[i] )

        ## 接着更新初始条件

        self.rho_S = rho_S
        self.rho_E = rho_E
        self.rho_I = rho_I
        self.rho_R = rho_R

        ## 更新流动概率C矩阵
        self.count_C()

        ## Estimate the initial probability of infection
        self.update_Πi()

        ## Initialize the output dataframe
        out_df = pd.DataFrame()
        #并将模型的起始日期转换为datetime格式

        self.start_date = startTime
        date = np.repeat(startTime, N, axis=0)

        out_df.loc[:, 'date'] = date
        out_df.loc[:, 'patch_id'] = patch_ids
        out_df.loc[:, 'rho_S'] = self.rho_S
        out_df.loc[:, 'rho_E'] = self.rho_E
        out_df.loc[:, 'rho_I'] = self.rho_I
        out_df.loc[:, 'rho_R'] = self.rho_R
        out_df.loc[:, 'Πi'] = self.Πi
        out_df.loc[:, 'Pi'] = self.Pi

        self.out_df = out_df

        return None

    #启动了MMCA模型，在该模型中，每天根据马尔可夫过程更新感染概率
    #马尔可夫过程反过来控制从易感个体到感染前阶段的转变。
    def run_epidemic_spreading(self, timeStep):

        ## #建议使用 mmca.set_seed(E_0, I_0) 来设置感染种子
        assert hasattr(self, 'rho_I'), "You must set the infectious seeds before running the model. Maybe try: mmca.set_seed(E_0, I_0)"
        

        ## 对模型进行时间步长的循环，从0到tau
        for t in range(timeStep):

            #打印当前时间步长
            # print("\n")
            # print("Timestep t: %s"%(t))

            # 更新下一天的各个隔室的计数
            self.compartment_evolution()

            # 更新感染概率Πi
            self.update_Πi()

            # 更新输出数据框
            self.update_output(t)

            # print("======================================")
 
        
        return None


    #克罗内克（Kronecker）函数用于表示两个指标是否相等，如果 i 等于 j，则函数返回1，否则返回0。
    def kronecker(self, i, j):
        return 1 if i == j else 0

    #计算Patch i中节点被感染的总概率
    def update_Πi(self):


        N = self.N
        p = self.p

        # C = self.C
        # print("输出C矩阵1:", C)

        Πi = np.zeros(N)
        Pi = np.zeros(N)
        
        # 计算节点仅在社区i被感染的概率
        Pi = self.update_Pi()
        # update_Pi()会更新一遍C
        C = self.C
        # print("输出C矩阵2:", C)

        #计算Patch i中节点被感染的总概率 ：在当前社区被感染的概率+移动到临近社区被感染的概率
        for i in range(N):
            sum_C = 0
            #在各个社区中的移动被感染的概率 ΣCij*Pj(t)
            for j in range(N):
                sum_C += C[i][j] *Pi[j]
                
            #计算Patch i中节点被感染的总概率
            Πi[i] = (1 - p) * Pi[i] + p * sum_C 

        self.Πi = Πi
        # print("Πi=", Πi)
        return Πi


    #计算从j转移到i的人数 n(j->i) moving_nji
    def count_nji(self, j, i):

        C = self.C
        p = self.p
        n_vec = self.n_vec


        #从j转移到i的人数 n(j->i) moving_nji
        moving_nji = self.kronecker(i, j)*(1-p)*n_vec[i] + p*C[j][i]*n_vec[j]

        return moving_nji
        
    #没有在patch i中被感染的人数占比 none_infected
    def update_Pi(self):

        N = self.N
        Lambda = self.Lambda
        rho_I = self.rho_I

        #更新移动C矩阵
        self.count_C()
        
        # print("更新后的移动矩阵C:",C)

        ## 计算Pi
        Pi = np.zeros(N)

        for i in range(N):

            #在patch i内未被感染的概率
            none_infected = 1

            ## 在每个源区中，对所有从区域 Σj 移动到区域 i 的感染人口进行求和
            for j in range(N):
                #计算指数部分
                exponent = self.count_nji(j, i)
                none_infected *= pow(1- Lambda*rho_I[j] , exponent)
                # print("从社区%d移动到%d的个体数量为:%.6f"%(j, i, exponent))

                # print("rho_I[%d]为:%.6f"%(j, rho_I[j]))
            # print("未感染的数量为:%.6f"%none_infected)
            Pi[i] = 1 - none_infected
            # print("Pi[%d]:%.6f"%(i, Pi[i]))
            # print("Pi[%d]=%.6f"%(i, Pi[i]))
            # print("\n")
        
        self.Pi = Pi

        return Pi
    
    #根据给定的传播模型和参数计算下一天的各个舱室的数量
    def compartment_evolution(self):

        ## 区域数量 
        N = self.N

        ## 初始化舱室
        rho_S = np.zeros(N)
        rho_E = np.zeros(N)
        rho_I = np.zeros(N)
        rho_R = np.zeros(N)

        ## 更新每个区域的舱室估计 compartment estimates 
        for i in range(N):
            #使用上一步计算得到的传播概率Pi来更新易感染（S）状态的数量。
            rho_S[i] = (1 - self.Πi[i]) * self.rho_S[i]
            rho_E[i] = self.Πi[i] * self.rho_S[i] + (1 - self.beta) * self.rho_E[i]
            rho_I[i] = self.beta * self.rho_E[i] + (1 - self.mu) * self.rho_I[i]
            rho_R[i] = self.rho_R[i] +  self.mu * self.rho_I[i] 
 
        self.rho_S = rho_S
        self.rho_E = rho_E
        self.rho_I = rho_I
        self.rho_R = rho_R
        #更新模型中各个状态的数量，并返回更新后的易感染（S）、暴露（E）、感染（I）、康复（R）
        return self.rho_S, self.rho_E, self.rho_I, self.rho_R

    #更新输出数据框，包括所有当前模型步骤的参数
    def update_output(self, t):
      
        #获取模型中的区域数量N
        N = self.N

        #计算当前日期，即开始日期加上当前时间步长的时间间隔，得到今天的日期
        today = t + 1
        #创建一个空的DataFrame来存储今天的数据
        today_df = pd.DataFrame()
        #创建一个日期列，将今天的日期重复N次，以匹配每个区域
        date = np.repeat(today, N, axis=0)

        today_df.loc[:, 'date'] = date
        today_df.loc[:, 'patch_id'] = self.patch_ids
        today_df.loc[:, 'rho_S'] = self.rho_S
        today_df.loc[:, 'rho_E'] = self.rho_E
        today_df.loc[:, 'rho_I'] = self.rho_I
        today_df.loc[:, 'rho_R'] = self.rho_R
        today_df.loc[:, 'Πi'] = self.Πi
        today_df.loc[:, 'Pi'] = self.Pi

        #将今天的DataFrame与之前的输出数据框合并
        self.out_df = pd.concat([self.out_df, today_df])

        return None

