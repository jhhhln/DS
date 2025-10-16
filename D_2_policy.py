import numpy as np
from demand import sample_generation
import itertools
import pandas as pd
from tqdm import tqdm

class dual_sourcing:
    def __init__(self, c_r, c_e, h, b, l_r, l_e, T, sample, service_level):
        #成本参数
        self.c_r = c_r
        self.c_e = c_e
        self.h   = h
        self.b   = b
        self.l_r = l_r
        self.l_e = l_e
        self.l=self.l_r-self.l_e
        #给定的服务水平alpha
        self.service_level = service_level
        self.T = T

        # 输入需求 [N, T] 其中列数代表一条路径的总周期，行数代表路径的总条数，用demand得到的cost作比较
        self.sample = sample
        self.mean = self.sample.mean()
        self.S_service_level = self.cum_demand_quantile()

        # # 加急初始订单
        self.q_init = np.zeros(self.l_e)
        self.x_init = np.diff(np.insert(self.S_service_level, 0, 0))[:self.l_e+1]
        self.x_init = np.concatenate((self.x_init,
                                      np.ones(self.l_r - self.l_e - 1) * self.mean))

        self.Se=self.S_service_level[self.l_e]
        self.Sr=self.S_service_level[self.l_r]
        self.num_search_range=100


    #计算S1,到S_(l_r+1)
    def cum_demand_quantile(self):
        #对需求数组先向右求和，然后对每一列分别取分位数，序号为k取的是(k+1)个D的情况
        q_all = np.quantile(self.sample.cumsum(axis=1), q=self.service_level, axis=0)  
        return q_all[:self.l_r + 1] 


    def cal_cost(self, order_record_r, order_record_e,  inv_level_record):
        period_cost = self.c_r * order_record_r \
            + self.c_e * order_record_e \
            + self.h * np.maximum(inv_level_record[:,1:], 0) \
            + self.b * np.maximum(-inv_level_record[:,1:], 0)
        average_total_cost = period_cost.sum(axis=1).mean()
        return period_cost, average_total_cost
    

    def single_source(self, demand):
        iter_num = demand.shape[0]
        period_length = demand.shape[1]

        #定义双源系统两个渠道的订货量
        order_record_regular=np.tile(self.x_init,(iter_num,1))
        order_record_expe=np.tile(self.q_init,(iter_num,1))

        # 初始化记录数组
        inv_level_record = np.zeros((iter_num, 1))
        S_r = self.Se + (self.l_r - self.l_e) * self.mean
    
        for t in range(period_length):
            if order_record_expe.shape[1] < period_length:
                IP_e = inv_level_record[:,[t]] \
                    + order_record_regular[:,t : t+self.l_e+1].sum(axis=1)[:,None] \
                    + order_record_expe[:,t : t+self.l_e].sum(axis=1)[:,None]
                order_e = np.maximum(np.ones(IP_e.shape) * self.Se - IP_e, 0)
                order_record_expe = np.hstack((order_record_expe, order_e))

                if order_record_regular.shape[1] < period_length:
                    IP_r = inv_level_record[:,[t]] + order_record_regular[:,t:t+self.l_r].sum(axis=1)[:,None]
                    order_r = np.maximum(np.ones(IP_r.shape) * S_r - IP_r, 0)
                    order_r = np.minimum(order_r - order_e, 0)
                    order_record_regular = np.hstack((order_record_regular,order_r))

            next_inv_level = inv_level_record[:, [t]] + order_record_regular[:, [t]] + order_record_expe[:, [t]] - demand[:, [t]]
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
        period_cost, average_total_cost = self.cal_cost(order_r, np.zeros((iter_num, period_length)),  inv_level_record)

        return {
            'order_record_r': order_record_regular,
            'order_record_e': order_record_expe,
            'inv_level_record': inv_level_record,
            'period_cost': period_cost,
            'average_total_cost': average_total_cost
        }


    # 含义是给定常规渠道和加急渠道的inventory position之后如何订货和每条路径的成本
    def cal_order_up_to(self, demand, S_e, S_r,
                        x_init, q_init,
                        constraint_D2=False):
        iter_num = demand.shape[0]
        period_length = demand.shape[1]

        #定义双源系统两个渠道的订货量
        order_record_regular=np.tile(x_init,(iter_num,1))
        order_record_expe=np.tile(q_init,(iter_num,1))

        # 初始化记录数组
        inv_level_record = np.zeros((iter_num, 1))
        overshoot_record=np.zeros((iter_num, 1))

        for t in range(period_length):
            if order_record_expe.shape[1] < period_length:
                IP_e = inv_level_record[:,[t]] + order_record_expe[:,t:t+self.l_e].sum(axis=1)[:,None] \
                    + order_record_regular[:,t:t+self.l_e+1].sum(axis=1)[:,None]
                order_e=np.maximum(np.ones(IP_e.shape) * S_e - IP_e, 0)
                if constraint_D2:
                    order_e=np.maximum(np.ones(IP_e.shape) * self.Se - IP_e, 0)
                order_record_expe=np.hstack((order_record_expe,order_e))
                overshoot=np.maximum(IP_e-np.ones(IP_e.shape) * S_e, 0)
                overshoot_record=np.hstack((overshoot_record,overshoot))
            
            if order_record_regular.shape[1] < period_length:
                IP_r = inv_level_record[:,[t]] + order_record_expe[:,t:t+self.l_e+1].sum(axis=1)[:,None] \
                    + order_record_regular[:,t:t+self.l_r].sum(axis=1)[:,None]
                order_r=np.maximum(np.ones(IP_r.shape) * S_r - IP_r, 0)
                order_record_regular=np.hstack((order_record_regular,order_r))

            next_inv_level = inv_level_record[:, [t]] + order_record_regular[:, [t]] \
                + order_record_expe[:,[t]] - demand[:, [t]]
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
        period_cost, average_total_cost = self.cal_cost(order_record_regular, order_record_expe,  inv_level_record)

        return {
            'order_record_r': order_record_regular,
            'order_record_e': order_record_expe,
            'inv_level_record': inv_level_record,
            'period_cost': period_cost,
            'average_total_cost': average_total_cost,
            'overshoot_record':overshoot_record
        }        


    def constrained_DI_policy(self, demand, sample, x_init=None,q_init=None):
        #先找到delta的稳态分布，然后给出可能最优的组合（S,S+delta),在组合中搜索最优成本
        #先利用sample path找到稳态分布
        x_init=self.x_init if x_init is None else x_init
        q_init=self.q_init if q_init is None else q_init

        delta_range = np.arange(0, self.Sr, self.Sr/self.num_search_range)

        DI_cost_record =[]
        for delta in delta_range:
            record=self.cal_order_up_to(sample, self.Sr-delta, self.Sr, 
                                        x_init, q_init,
                                        constraint_D2=True)
            DI_cost_record.append(record['average_total_cost'])
        min_cost_idx = np.argmin(DI_cost_record)
        optimal_delta = delta_range[min_cost_idx]

        record_of_demand=self.cal_order_up_to(demand, self.Sr-optimal_delta, self.Sr,
                                              x_init, q_init,
                                              constraint_D2=True)
        return record_of_demand
    

    def benchmark_DI_policy(self, demand, sample, x_init=None,q_init=None):
        #先找到delta的稳态分布，然后给出可能最优的组合（S,S+delta),在组合中搜索最优成本
        #先利用sample path找到稳态分布
        x_init=self.x_init if x_init is None else x_init
        q_init=self.q_init if q_init is None else q_init

        delta_range = np.arange(0, self.Sr, self.Sr/self.num_search_range)

        DI_cost_record =[]
        best_Se_record=[]
        for delta in delta_range:
            record=self.cal_order_up_to(sample,0,delta,x_init,q_init,constraint_D2=False)
            overshoot_record=record['overshoot_record'] 
            #计算最优的Se
            variable = sample.cumsum(axis=1)[:, self.l_e][:, None].reshape(1,-1)
            best_Se = np.quantile(
                variable - overshoot_record[:, -1], 
                self.b / (self.b + self.h))

            result = self.cal_order_up_to(sample, best_Se, delta+best_Se, x_init, q_init, constraint_D2=False)
            DI_cost_record.append(result['average_total_cost'])
            best_Se_record.append(best_Se)

        min_cost_idx = np.argmin(DI_cost_record)
        optimal_delta = delta_range[min_cost_idx]
        optimal_Se = best_Se_record[min_cost_idx]
        record_of_demand=self.cal_order_up_to(demand, optimal_Se, optimal_Se + optimal_delta,
                                              x_init, q_init,
                                              constraint_D2=True)
        return record_of_demand


    # ATBS policy implementation with order-up-to level S_e and fixed regular order r
    def cal_order_up_to_with_r(self, demand, S_e,r, x_init, q_init, constraint_D2=False):
        iter_num = demand.shape[0]
        period_length = demand.shape[1]

        #定义双源系统两个渠道的订货量
        order_record_regular=np.tile(x_init,(iter_num,1))
        order_record_expe=np.tile(q_init,(iter_num,1))

        # 初始化记录数组
        inv_level_record = np.zeros((iter_num, 1))
        overshoot_record=np.zeros((iter_num, 1))

        for t in range(period_length):
            if order_record_expe.shape[1] < period_length:
                IP_e=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e].sum(axis=1)[:,None]+order_record_regular[:,t:t+self.l_e+1].sum(axis=1)[:,None]
                ###
                order_e=np.maximum(np.ones(IP_e.shape) * S_e - IP_e, 0)
                if constraint_D2:
                    order_e=np.maximum(np.ones(IP_e.shape) * self.Se - IP_e, 0)
                order_record_expe=np.hstack((order_record_expe,order_e))
                overshoot=np.maximum(IP_e-np.ones(IP_e.shape) * S_e, 0)
                overshoot_record=np.hstack((overshoot_record,overshoot))

            if order_record_regular.shape[1] < period_length:
                order_record_regular=np.hstack((order_record_regular,
                                                np.ones((iter_num, 1))*r))
                
            next_inv_level = inv_level_record[:, [t]] + order_record_regular[:, [t]] \
                + order_record_expe[:,[t]] - demand[:, [t]]
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
        period_cost, average_total_cost = self.cal_cost(order_record_regular, order_record_expe,  inv_level_record)

        return {
            'order_record_r': order_record_regular,
            'order_record_e': order_record_expe,
            'inv_level_record': inv_level_record,
            'period_cost': period_cost,
            'average_total_cost': average_total_cost,
            'overshoot_record':overshoot_record
        }          

    def constrained_TBS_policy(self, sample, demand, x_init=None, q_init=None):
        x_init = self.x_init if x_init is None else x_init
        q_init = self.q_init if q_init is None else q_init
        r_range = np.linspace(0,self.mean, self.num_search_range)

        TBS_cost_record = []      
        best_Se_record  = []     

        for r in r_range:
            #首先根据根据稳态对于给定的常规渠道订货量r搜索对应的最优SE
            record = self.cal_order_up_to_with_r(sample, self.Se, r,
                                                x_init, q_init,
                                                constraint_D2=False)
            overshoot_record = record['overshoot_record']
            # 计算 best_Se
            variable = sample.cumsum(axis=1)[:, self.l_e][:, None].reshape(1,-1)
            best_Se = np.quantile(
                variable - overshoot_record[:, -1], 
                self.b / (self.b + self.h))

            #对于参数对(r,Se)考虑服务水平
            result = self.cal_order_up_to_with_r(sample, best_Se, r,
                                                 x_init, q_init,
                                                 constraint_D2=True)
            TBS_cost_record.append(result['average_total_cost'])
            best_Se_record.append(best_Se)

        min_cost_idx = np.argmin(TBS_cost_record)
        optimal_r = r_range[min_cost_idx]
        optimal_Se = best_Se_record[min_cost_idx]

        # 最优记录
        optimal_record = self.cal_order_up_to_with_r(
            demand, optimal_Se, optimal_r,
            x_init, q_init,
            constraint_D2=True
        )
        return optimal_record
    
    def cost_driven_TBS_policy(self, sample, demand, x_init=None, q_init=None):
        x_init = self.x_init if x_init is None else x_init
        q_init = self.q_init if q_init is None else q_init
        r_range = np.linspace(0, self.mean, self.num_search_range)

        TBS_cost_record = []      
        best_Se_record  = []     

        for r in r_range:
            #首先根据根据稳态对于给定的常规渠道订货量r搜索对应的最优SE
            record = self.cal_order_up_to_with_r(sample, self.Se, r,
                                                x_init, q_init,
                                                constraint_D2=False)
            overshoot_record = record['overshoot_record']
            # 计算 best_Se
            variable = sample.cumsum(axis=1)[:, self.l_e][:, None].reshape(1,-1)
            best_Se = np.quantile(
                variable - overshoot_record[:, -1], 
                self.b / (self.b + self.h))

            #对于参数对(r,Se)考虑服务水平
            result = self.cal_order_up_to_with_r(sample, best_Se, r,
                                                 x_init, q_init,
                                                 constraint_D2=False)
            TBS_cost_record.append(result['average_total_cost'])
            best_Se_record.append(best_Se)

        min_cost_idx = np.argmin(TBS_cost_record)
        optimal_r = r_range[min_cost_idx]
        optimal_Se = best_Se_record[min_cost_idx]

        # 最优记录
        optimal_record = self.cal_order_up_to_with_r(
            demand, optimal_Se, optimal_r,
            x_init, q_init,
            constraint_D2=True
        )
        return optimal_record

    def cal_fill_rate(self, demand, result_record_dict):
        #对于每条路径的每个节点，计算在到达时刻的总库存是否能满足需求
        #相当于可以得到一个[N,T]数组，每一个点都有若干个0-1变量表达是否满足(对应着若干条路径)，然后对每个点都可以得到一个概率，最后对每一列求平均
        num_of_iter = demand.shape[0]
        
        order_record_r = result_record_dict['order_record_r']
        order_record_e = result_record_dict['order_record_e']
        inv_level_record = result_record_dict['inv_level_record']
        period_length = inv_level_record.shape[1]
        
        #service_level_Sr = []
        service_level_Se = []
        
        # 对于每个时间点t，使用多条路径来检验服务水平
        for t in range(period_length - self.l_e-1):
            # 为当前时间点t创建存储服务水平的数组
            #t_service_level_Sr = []
            t_service_level_Se = []

            # 对每条样本路径进行检查
            for sample in range(num_of_iter):
                current_future_demand_path = demand[sample, t: t+self.l_e+1]
                current_pipeline_r=order_record_r[:, t: t+self.l_e+1].sum(axis=1)[:,None]
                current_pipeline_e=order_record_e[:, t: t + self.l_e + 1].sum(axis=1)[:,None]
                current_inventory_level=inv_level_record[:, t][:,None]
                inventory_final=current_inventory_level+current_pipeline_r+current_pipeline_e-current_future_demand_path.sum()
                #计算inventory_final大于0的比例
                t_service_level_Se.append((inventory_final> -0.01).mean())
                
            # 对当前时间点t的所有样本路径求平均
            service_level_Se.append(np.mean(t_service_level_Se))
            # service_level_Se.append(np.mean(t_service_level_Se))
        
        return np.array(service_level_Se)    

if __name__ == "__main__":
    # 设置参数
    c_r = 0    # 常规订单成本
    c_e = 2   # 加急订单成本
    h = 1      # 库存持有成本

    l_r = 10 # 常规订单提前期
    l_e = 3   # 加急订单提前期
    b = c_e+h*(l_r+1)    # 缺货成本
    T = 90   # 时间周期数
    N = 500  # 模拟路径数量
    N_1 = 1000
    service_level = 0.99  # 服务水平
    
    # 生成需求数据 - 使用正态分布
    distribution = ('norm', (100, 10))  # 均值为10，标准差为10的正态分布
    mean = distribution[1][0] 
    demand = sample_generation(distribution, (N, T))
    sample= sample_generation(distribution, (N_1, 500))
    sample2= sample_generation(distribution, (1000, 500))
    
    # 创建 dual_sourcing 实例
    print('single_cource')
    ds = dual_sourcing(c_r, c_e, h, b, l_r, l_e, T, sample2, service_level)
    single_source_result=ds.single_source(demand)
    print(single_source_result['average_total_cost'])
    print(ds.cal_fill_rate(sample2, single_source_result))


    #调用TBS策略
    print('TBS')
    TBS_result=ds.constrained_TBS_policy(sample,demand)
    print(TBS_result['average_total_cost'])
    print(ds.cal_fill_rate(sample2, TBS_result))
    
    print('benchmark DI')
    benchmark_di_result = ds.benchmark_DI_policy(demand, sample)
    print(benchmark_di_result['average_total_cost'])
    print(ds.cal_fill_rate(sample2, benchmark_di_result))

    print('cost driven TBS')
    cost_driven_TBS_result=ds.cost_driven_TBS_policy(sample,demand)
    print(cost_driven_TBS_result['average_total_cost'])
    print(ds.cal_fill_rate(sample2, cost_driven_TBS_result))
    print(1)

    # aa = single_source_result['order_record_r'][:,:90].cumsum(axis=1) + single_source_result['order_record_e'][:,:90].cumsum(axis=1)
    # bb = cost_driven_TBS_result['order_record_r'].cumsum(axis=1) + cost_driven_TBS_result['order_record_e'].cumsum(axis=1)
    # np.maximum(single_source_result['inv_level_record'], 0).sum(axis=1).mean() - np.maximum(cost_driven_TBS_result['inv_level_record'], 0).sum(axis=1).mean()
    for t in range(single_source_result['inv_level_record'].shape[1] - l_e-1):
        IP_e = single_source_result['inv_level_record'][:,[t]] + single_source_result['order_record_e'][:,t:t+l_e+1].sum(axis=1)[:,None] \
                    + single_source_result['order_record_r'][:,t:t+l_e+1].sum(axis=1)[:,None]
        if (IP_e < ds.Se - 0.001).any():
            print(t, np.where(IP_e < ds.Se)[0])