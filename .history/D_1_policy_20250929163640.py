import numpy as np
from demand import sample_generation
 
class dual_sourcing:
    def __init__(self, c_r, c_e, h, b, l_r, l_e, demand, service_level):

        self.c_r = c_r
        self.c_e = c_e
        self.h   = h
        self.b   = b
        self.l_r = l_r
        self.l_e = l_e
        self.l=self.l_r-self.l_e
        #给定的服务水平alpha
        self.service_level = service_level

        # 输入需求 [N, T] 其中列数代表一条路径的总周期，行数代表路径的总条数，用demand得到的cost作比较
        self.demand = demand
        #对数组的需求向右累加求和 每一列代表k个D的可能的取值，之后对每一列分别取分位数
        self.cum_demand = self.demand.cumsum(axis=1)

        # S_1,…,S_{l_r+1}
        self.S_service_level = self.cum_demand_quantile()

        # 加急初始订单
        self.q_init = np.zeros(self.l_e)

        # 常规初始订单 lost_sales和DDI
        self.x_init_DDI = np.diff(np.insert(self.S_service_level, 0, 0))[:self.l_r]

        # init x for SDI
        #前l_e+1项
        self.x_init_SDI=np.diff(np.insert(self.S_service_level, 0, 0))[:self.l_e+1]
        #第l_e+2到第l_r项
        S_l=[]
        for i in range(self.l-1):
            S_l.append(self.Sl(0, self.c_e - self.c_r + self.l*self.h, 
                            self.h, self.b , i+1, 1-self.service_level))     
        x_init_SDI_add=np.diff(np.insert(S_l,0,0))[:self.l-1]
        self.x_init_SDI = np.concatenate((self.x_init_SDI,x_init_SDI_add))


        self.Se=self.S_service_level[self.l_e]
        self.Sr=self.S_service_level[self.l_r]
        self.S_l=self.Sl(0, self.c_e - self.c_r + self.l *self.h, self.h, self.b, self.l, 1-self.service_level)
        
        self.num_search_range=100


#计算S1,到S_(l_r+1)
    def cum_demand_quantile(self):
        #对需求数组先向右求和，然后对每一列分别取分位数，序号为k取的是(k+1)个D的情况
        q_all = np.quantile(self.cum_demand,
                            q=self.service_level,
                            axis=0)  
        return q_all[:self.l_r + 1]

#计算S_l函数
    def Sl(self,c1,c2,h,b,l,alpha):
        quantile = (b*alpha + c2 - c1) / (h*(1-alpha) + b*alpha + c2 - c1)
        return np.quantile(self.cum_demand[:,l-1], quantile)

    #基于lost-sales系统构造DDI策略 这里的S一般应该是self.Sr
    def lost_sales(self, demand,S=None,inventory_level=0):
        S=self.Sr if S is None else S
        #一共模拟的path次数N
        iter_num = demand.shape[0]
        #每条path的长度T
        period_length = demand.shape[1]
        #希望记录每一期的期初库存y,每一期的订货量order以及每一期的成本
        #对于每一条路径都赋予相同的初始订货量，*N
        x_init = np.tile(self.x_init_DDI, (iter_num, 1))

        order_record = x_init.copy()
        inv_level_record = np.ones((iter_num, 1)) * inventory_level
        cost_per_period = np.zeros((iter_num, 0)) 
        y_level_record = np.zeros((iter_num, 1))

        for t in range(period_length):
            #下订单的过程只需要进行T-l_r期，因为之后下的订单不会在我们考虑的T期到达
            if order_record.shape[1] < period_length:
                
                # 计算订单 第一期开始之后下的订单就需要满足order_up_to策略了
                x_order = np.maximum(np.ones((iter_num, 1)) * S- order_record[:, t:t + self.l_r].sum(axis=1)[:, None]- inv_level_record[:, [t]],0)
                # 使用 hstack 添加列
                order_record = np.hstack((order_record, x_order))

            # 计算订单到达之后的本期手头库存(起初+到达)和本期的需求d
            y = inv_level_record[:, [t]] + order_record[:, [t]]
            d = demand[:, [t]]
            # 计算当前周期的成本
            period_cost = (self.c_r * order_record[:, [t]]+ self.h * np.maximum(y - d, 0)+ self.b * np.maximum(d - y, 0))
            # 将当前周期的成本添加到成本记录中
            cost_per_period = np.hstack((cost_per_period, period_cost))

            # 计算下一期的库存水平 lost_sales系统
            next_inv_level = np.maximum(y - d, 0)

            # 使用 hstack 添加列
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
            y_level_record = np.hstack((y_level_record, y))


        # 计算每个迭代（行）的总成本
        total_cost_per_iteration = np.sum(cost_per_period, axis=1)
        average_total_cost = np.mean(total_cost_per_iteration)


        return {
            'order_record': order_record,
            'inv_level_record': inv_level_record,
            'y_level_record': y_level_record,
            'cost_per_period': cost_per_period,
            'total_cost_per_iteration': total_cost_per_iteration,
            'average_total_cost': average_total_cost
        }
    def single_lost_sales(self, demand, S=None, inventory_level=0):
        S = self.Sr if S is None else S
        iter_num = demand.shape[0]
        period_length = demand.shape[1]
        
        x_init = np.tile(self.x_init_DDI, (iter_num, 1))
        order_record = x_init.copy()
        inv_level_record = np.ones((iter_num, 1)) * inventory_level
        cost_per_period = np.zeros((iter_num, 0)) 
        y_level_record = np.zeros((iter_num, 1))


        for t in range(period_length):
            if order_record.shape[1] < period_length:
                # end_idx = min(t + self.l_r, order_record.shape[1])
                x_order = np.maximum(np.ones((iter_num, 1)) * S- order_record[:, t:t + self.l_r].sum(axis=1)[:, None]- inv_level_record[:, [t]], 0)
                order_record = np.hstack((order_record, x_order))

            y = inv_level_record[:, [t]] + order_record[:, [t]]
            d = demand[:, [t]]
            
            next_inv_level = y - d 
            
            holding_cost = self.h * np.maximum(next_inv_level, 0)
            backlog_cost = self.b *np.maximum(-next_inv_level, 0)
            
            period_cost = (self.c_r * order_record[:, [t]] + holding_cost + backlog_cost)
            
            cost_per_period = np.hstack((cost_per_period, period_cost))

            inv_level_record = np.hstack((inv_level_record, next_inv_level))
            y_level_record = np.hstack((y_level_record, y))

        total_cost_per_iteration = np.sum(cost_per_period, axis=1)
        average_total_cost = np.mean(total_cost_per_iteration)

        return {
            'order_record': order_record,
            'inv_level_record': inv_level_record,
            'y_level_record': y_level_record,
            'cost_per_period': cost_per_period,
            'total_cost_per_iteration': total_cost_per_iteration,
            'average_total_cost': average_total_cost
        }


    def DDI_policy(self,demand,Se=None,D_2_constraint=False,inventory_level=0):
        Se=self.Se if Se is None else Se
        lost_sales_record=self.lost_sales(demand=demand,S=None,inventory_level=0)

        iter_num = demand.shape[0]
        period_length = demand.shape[1]
        
        #定义双源系统的两个渠道的订货量，每一条路径都赋予相同的初始订货量
        x_init = np.tile(self.x_init_DDI, (iter_num, 1))
        q_init= np.tile(self.q_init,(iter_num,1))
        #这里记录两个渠道的订货量
        order_record_regular = x_init.copy()
        order_record_expe=q_init.copy()

        # 初始化记录数组
        inv_level_record = np.ones((iter_num, 1)) * inventory_level
        cost_per_period = np.zeros((iter_num, 0)) 
        y_level_record = np.zeros((iter_num, 1))

        for t in range(period_length):
            #对于慢渠道来说，下订单的过程只需要进行T-l_r期，之后下的订单不会在T期内到达
            if order_record_regular.shape[1] < period_length:
                # 确保索引不越界
                end_idx = min(t + self.l_r, order_record_regular.shape[1])
                #对于t>=0之后的每一期下达的订单都是和lost_sales中的结果相同
                order_regular = lost_sales_record['order_record'][:, t+self.l_r][:, None]
                order_record_regular = np.hstack((order_record_regular, order_regular))
            #对于快渠道来说，下订单的过程只需要进行T-l_e期，之后下的订单也不会在T期内到达
            if order_record_expe.shape[1] < period_length:
                # 确保索引不越界
                end_idx = min(t + self.l_e, order_record_expe.shape[1])
                #根据更新的最新版策略，前delta l+1期的下单的加急渠道订货量均为0
                if t<self.l+1:
                    order_e=np.zeros((iter_num, 1))
                    order_record_expe=np.hstack((order_record_expe,order_e))
                else:
                    #初始的y为0 不用取出
                    # y_inv=lost_sales_record['y_level_record'][:, 1:]
                    # net_inv = y_inv - demand
                    # #对于每一条path的每一期，求出对应时间点的backlog值
                    # BACK = -np.minimum(net_inv, 0)
                    BACK=np.maximum(0,demand[:,:t]-lost_sales_record['order_record'][:,:t])
                    #计算每个时间点之前（包括当前时间点）的最大 BACK 值
                    cumulative_max = np.maximum.accumulate(BACK, axis=1)
                    if t==self.l+1:
                        order_e=cumulative_max[:,t-self.l-1][:,None]

                    else:
                        order_e=(cumulative_max[:,t-self.l-1] - cumulative_max[:,t-self.l-2])[:,None]
                    order_record_expe=np.hstack((order_record_expe,order_e))
                if D_2_constraint:
                    IP_e=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e+1].sum(axis=1)[:,None]+order_record_regular[:,t:t+self.l_e+1].sum(axis=1)[:,None]
                    order_e_add=np.maximum(np.ones(IP_e.shape) * Se - IP_e, 0)
                    order_record_expe[:, -1:] = order_record_expe[:, -1:] + order_e_add
                
            y = inv_level_record[:, [t]] + order_record_regular[:, [t]]+order_record_expe[:,[t]]
            d=demand[:,[t]]

            # 计算当前周期的成本
            period_cost = (
                self.c_r * order_record_regular[:, [t]]+self.c_e * order_record_expe[:, [t]]
                + self.h * np.maximum(y - d, 0)
                + self.b * np.maximum(d - y, 0)
            )

            
            # 将当前周期的成本添加到成本记录中
            cost_per_period = np.hstack((cost_per_period, period_cost))


            # 计算下一期的库存水平
            next_inv_level = y - d

            
            # 使用 hstack 添加列
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
            y_level_record = np.hstack((y_level_record, y))


        # 计算每个迭代（行）的总成本
        total_cost_per_iteration = np.sum(cost_per_period, axis=1)
        average_total_cost = np.mean(total_cost_per_iteration)
        print(f"平均总成本: {average_total_cost}")

        return {
            'order_record_r': order_record_regular,
            'order_record_e': order_record_expe,
            'inv_level_record': inv_level_record,
            'y_level_record': y_level_record,
            'cost_per_period': cost_per_period,
            'total_cost_per_iteration': total_cost_per_iteration,
            'average_total_cost': average_total_cost
        }


    def SDI_policy(self,demand,inventory_level=0):

        iter_num = demand.shape[0]
        period_length = demand.shape[1]

        #定义双源系统两个渠道的订货量
        x_init=np.tile(self.x_init_SDI,(iter_num,1))
        q_init=np.tile(self.q_init,(iter_num,1))
        #记录两个渠道的订货量
        order_record_regular = x_init.copy()
        order_record_expe=q_init.copy()

        # 初始化记录数组
        inv_level_record = np.ones((iter_num, 1)) * inventory_level
        cost_per_period = np.zeros((iter_num, 0)) 
        y_level_record = np.zeros((iter_num, 1))
        
        for t in range(period_length):
            if order_record_expe.shape[1] < period_length:
                # 确保索引不越界
                IP_e=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e].sum(axis=1)[:,None]
                +order_record_regular[:,t:t+self.l_e+1].sum(axis=1)[:,None]
                order_e=np.maximum(np.ones(IP_e.shape) * self.Se - IP_e, 0)
                order_record_expe=np.hstack((order_record_expe,order_e))
            
            if order_record_regular.shape[1] < period_length:
                # 确保索引不越界
                end_idx = min(t + self.l_r, order_record_regular.shape[1])
                IP_r=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e+1].sum(axis=1)[:,None]
                +order_record_regular[:,t:t+self.l_r].sum(axis=1)[:,None]
                order_r=np.maximum(np.ones(IP_r.shape) * (self.Se+self.S_l) - IP_r, 0)
                order_record_regular=np.hstack((order_record_regular,order_r))

            y = inv_level_record[:, [t]] + order_record_regular[:, [t]]+order_record_expe[:,[t]]
            d=demand[:,[t]]

            # 计算当前周期的成本
            period_cost = (
                self.c_r * order_record_regular[:, [t]]+self.c_e * order_record_expe[:, [t]]
                + self.h * np.maximum(y - d, 0)
                + self.b * np.maximum(d - y, 0)
            )

            
            # 将当前周期的成本添加到成本记录中
            cost_per_period = np.hstack((cost_per_period, period_cost))


            # 计算下一期的库存水平
            next_inv_level = y - d

            
            # 使用 hstack 添加列
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
            y_level_record = np.hstack((y_level_record, y))


        # 计算每个迭代（行）的总成本
        total_cost_per_iteration = np.sum(cost_per_period, axis=1)
        average_total_cost = np.mean(total_cost_per_iteration)
        print(f"平均总成本: {average_total_cost}")

        return {
            'order_record_r': order_record_regular,
            'order_record_e': order_record_expe,
            'inv_level_record': inv_level_record,
            'y_level_record': y_level_record,
            'cost_per_period': cost_per_period,
            'total_cost_per_iteration': total_cost_per_iteration,
            'average_total_cost': average_total_cost
        }       

#含义是给定常规渠道和加急渠道的inventory position之后如何订货和每条路径的成本
    def cal_order_up_to(self,demand,S_e,S_r,x_init,q_init,inventory_level=0):

        iter_num = demand.shape[0]
        period_length = demand.shape[1]

        #定义双源系统两个渠道的订货量
        x_init=np.tile(x_init,(iter_num,1))
        q_init=np.tile(q_init,(iter_num,1))
        #记录两个渠道的订货量
        order_record_regular = x_init.copy()
        order_record_expe=q_init.copy()

        # 初始化记录数组
        inv_level_record = np.ones((iter_num, 1)) * inventory_level
        cost_per_period = np.zeros((iter_num, 0)) 
        y_level_record = np.zeros((iter_num, 1))
        overshoot_record=np.zeros((iter_num, 1))

        for t in range(period_length):
            if order_record_expe.shape[1] < period_length:
                # 确保索引不越界
                end_idx = min(t + self.l_e, order_record_expe.shape[1])
                IP_e=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e].sum(axis=1)[:,None]+order_record_regular[:,t:t+self.l_e+1].sum(axis=1)[:,None]
                order_e=np.maximum(np.ones(IP_e.shape) * S_e - IP_e, 0)
                order_record_expe=np.hstack((order_record_expe,order_e))
                overshoot=np.maximum(IP_e-np.ones(IP_e.shape) * S_e, 0)
                overshoot_record=np.hstack((overshoot_record,overshoot))
            
            if order_record_regular.shape[1] < period_length:
                # 确保索引不越界
                end_idx = min(t + self.l_r, order_record_regular.shape[1])
                IP_r=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e+1].sum(axis=1)[:,None]+order_record_regular[:,t:t+self.l_r].sum(axis=1)[:,None]
                order_r=np.maximum(np.ones(IP_r.shape) * S_r - IP_r, 0)
                order_record_regular=np.hstack((order_record_regular,order_r))

            y = inv_level_record[:, [t]] + order_record_regular[:, [t]]+order_record_expe[:,[t]]
            d=demand[:,[t]]

            # 计算当前周期的成本
            period_cost = (
                self.c_r * order_record_regular[:, [t]]+self.c_e * order_record_expe[:, [t]]
                + self.h * np.maximum(y - d, 0)
                + self.b * np.maximum(d - y, 0)
            )

            
            # 将当前周期的成本添加到成本记录中
            cost_per_period = np.hstack((cost_per_period, period_cost))


            # 计算下一期的库存水平
            next_inv_level = y - d

            
            # 使用 hstack 添加列
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
            y_level_record = np.hstack((y_level_record, y))


        # 计算每个迭代（行）的总成本
        total_cost_per_iteration = np.sum(cost_per_period, axis=1)
        average_total_cost = np.mean(total_cost_per_iteration)

        return {
            'order_record_r': order_record_regular,
            'order_record_e': order_record_expe,
            'inv_level_record': inv_level_record,
            'y_level_record': y_level_record,
            'cost_per_period': cost_per_period,
            'total_cost_per_iteration': total_cost_per_iteration,
            'average_total_cost': average_total_cost,
            'overshoot_record':overshoot_record
        }       


    def DI_policy(self,demand,sample,x_init=None,q_init=None,inventory_level=0):
        #先找到delta的稳态分布，然后给出可能最优的组合（S_e,S_e+delta),在组合中搜索最优成本
        #先利用sample path找到稳态分布
        x_init=self.x_init_DDI if x_init is None else x_init
        q_init=self.q_init if q_init is None else q_init

        delta_range = np.arange(0, self.Sr, self.num_search_range)

        DI_cost_record =[]
        for delta in delta_range:
            record=self.cal_order_up_to(sample,self.Sr-delta,self.Sr,x_init,q_init,inventory_level=0)
           
            cost=record['average_total_cost']
            DI_cost_record.append(cost)
        min_cost_idx = np.argmin(DI_cost_record)
        optimal_delta = delta_range[min_cost_idx]


        record_of_demand=self.cal_order_up_to(demand,self.Sr-optimal_delta,self.Sr,x_init,q_init,inventory_level=0)
        return record_of_demand
#TBS策略中加急渠道是order_up_to,常规渠道是r
    def cal_order_up_to_with_r(self,demand,S_e,r,x_init,q_init,inventory_level=0,constraint_D1=False):

        iter_num = demand.shape[0]
        period_length = demand.shape[1]

        #定义双源系统两个渠道的订货量
        x_init=np.tile(x_init,(iter_num,1))
        q_init=np.tile(q_init,(iter_num,1))
        #记录两个渠道的订货量
        order_record_regular = x_init.copy()
        order_record_expe=q_init.copy()

        # 初始化记录数组
        inv_level_record = np.ones((iter_num, 1)) * inventory_level
        cost_per_period = np.zeros((iter_num, 0)) 
        y_level_record = np.zeros((iter_num, 1))
        overshoot_record=np.zeros((iter_num, 1))

        for t in range(period_length):
            if order_record_expe.shape[1] < period_length:
                # 确保索引不越界
                end_idx = min(t + self.l_e, order_record_expe.shape[1])
                IP_e=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e].sum(axis=1)[:,None]+order_record_regular[:,t:t+self.l_e+1].sum(axis=1)[:,None]
                order_e=np.maximum(np.ones(IP_e.shape) * S_e - IP_e, 0)
                order_record_expe=np.hstack((order_record_expe,order_e))
                overshoot=np.maximum(IP_e-np.ones(IP_e.shape) * S_e, 0)
                overshoot_record=np.hstack((overshoot_record,overshoot))
            if order_record_regular.shape[1] < period_length:
                # 确保索引不越界
                end_idx = min(t + self.l_r, order_record_regular.shape[1])
                IP_r=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e+1].sum(axis=1)[:,None]\
                +order_record_regular[:,t:t+self.l_r].sum(axis=1)[:,None]
                order_r=np.ones(IP_r.shape)*r
                order_record_regular=np.hstack((order_record_regular,order_r))
                if constraint_D1:
                    IP_r=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e+1].sum(axis=1)[:,None]\
                    +order_record_regular[:,t:t+self.l_r+1].sum(axis=1)[:,None]
                    order_r_add=np.maximum(self.Sr - IP_r, 0)
                    order_record_regular[:, -1:] = order_record_regular[:, -1:] + order_r_add
                
            # overshoot=np.maximum(overshoot_record[:,-1:]-demand[:,t]+order_record_regular[:, -1:], 0)
            # overshoot_record=np.hstack((overshoot_record,overshoot))



            y = inv_level_record[:, [t]] + order_record_regular[:, [t]]+order_record_expe[:,[t]]
            d=demand[:,[t]]

            # 计算当前周期的成本
            period_cost = (
                self.c_r * order_record_regular[:, [t]]+self.c_e * order_record_expe[:, [t]]
                + self.h * np.maximum(y - d, 0)
                + self.b * np.maximum(d - y, 0)
            )

            
            # 将当前周期的成本添加到成本记录中
            cost_per_period = np.hstack((cost_per_period, period_cost))


            # 计算下一期的库存水平
            next_inv_level = y - d

            
            # 使用 hstack 添加列
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
            y_level_record = np.hstack((y_level_record, y))


        # 计算每个迭代（行）的总成本
        total_cost_per_iteration = np.sum(cost_per_period, axis=1)
        average_total_cost = np.mean(total_cost_per_iteration)

        return {
            'order_record_r': order_record_regular,
            'order_record_e': order_record_expe,
            'inv_level_record': inv_level_record,
            'y_level_record': y_level_record,
            'cost_per_period': cost_per_period,
            'total_cost_per_iteration': total_cost_per_iteration,
            'average_total_cost': average_total_cost,
            'overshoot_record':overshoot_record
        }       

    def TBS_policy(self,sample,demand,mean,x_init=None,q_init=None):
        x_init=self.x_init_DDI if x_init is None else x_init
        q_init=self.q_init if q_init is None else q_init
        r_range = np.linspace(0,mean, self.num_search_range)
        TBS_cost_record =[]
        best_se=[]
        for r in r_range:
            #给定r和SE的情况下计算成本，但有可能不满足D_1约束
            record=self.cal_order_up_to_with_r(sample,self.Se,r,x_init,q_init,inventory_level=0,constraint_D1=False)
            overshoot_record=record['overshoot_record']
            iter_num = overshoot_record.shape[0]
            #提取l_e+1个D相加之后的所有可能值，结果形式是N*1维的
            variable=sample.cumsum(axis=1)[:,self.l_e][:,None]
            #想把variable由N*1维变成N*50维的
            variable = np.tile(variable, (1, 100))
            best_Se=np.quantile(variable-overshoot_record[:,-100:],self.b/(self.b+self.h))
            best_se.append(best_Se)
            cost=self.cal_order_up_to_with_r(demand,best_Se,r,x_init,q_init,inventory_level=0,constraint_D1=False)['average_total_cost']
            TBS_cost_record.append(cost)
        min_cost_idx = np.argmin(TBS_cost_record)
        best_r=r_range[min_cost_idx]
        best_Se=best_se[min_cost_idx]
        optimal_record=self.cal_order_up_to_with_r(demand,best_Se,best_r,x_init,q_init,inventory_level=0,constraint_D1=True)
        return optimal_record    
            
    def cal_fill_rate(self, demand, result_record_dict):
        #对于每条路径的每个节点，计算在到达时刻的总库存是否能满足需求
        #相当于可以得到一个[N,T]数组，每一个点都有若干个0-1变量表达是否满足(对应着若干条路径)，然后对每个点都可以得到一个概率，最后对每一列求平均
        num_of_iter = demand.shape[0]
        
        order_record_r = result_record_dict['order_record_r']
        order_record_e = result_record_dict['order_record_e']
        inv_level_record = result_record_dict['inv_level_record']
        period_length = inv_level_record.shape[1]
        
        service_level_Sr = []
        #service_level_Se = []
        
        # 对于每个时间点t，使用多条路径来检验服务水平
        for t in range(period_length - self.l_r):
            # 为当前时间点t创建存储服务水平的数组
            t_service_level_Sr = []
            # t_service_level_Se = []

            # 对每条样本路径进行检查
            for sample in range(num_of_iter):
                current_future_demand_path = demand[sample, t: t+self.l_r+1]
                current_pipeline_r=order_record_r[:, t: t+self.l_r+1].sum(axis=1)[:,None]
                current_pipeline_e=order_record_e[:, t: t + self.l_e + 1].sum(axis=1)[:,None]
                current_inventory_level=inv_level_record[:, t][:,None]
                inventory_final=current_inventory_level+current_pipeline_r+current_pipeline_e-current_future_demand_path.sum()
                #计算inventory_final大于0的比例
                t_service_level_Sr.append((inventory_final> 0).mean())
                
            
            # 对当前时间点t的所有样本路径求平均
            service_level_Sr.append(np.mean(t_service_level_Sr))
            # service_level_Se.append(np.mean(t_service_level_Se))
        
        return np.array(service_level_Sr)
            
if __name__ == "__main__":
    # 设置参数
    c_r = 0    # 常规订单成本
    c_e = 10   # 加急订单成本
    h = 1      # 库存持有成本

    l_r = 3  # 常规订单提前期
    l_e = 1    # 加急订单提前期
    b = c_e+h*(l_r+1)    # 缺货成本
    T = 30   # 时间周期数
    N = 500  # 模拟路径数量
    service_level = 0.95  # 服务水平
    N_1=100
    
    # 生成需求数据 - 使用正态分布
    distribution = ('norm', (100, 10))  # 均值为10，标准差为10的正态分布
    demand = sample_generation(distribution, (N, T), random_seed=36)
    sample= sample_generation(distribution, (N_1, 1000), random_seed=42)
    
    # 创建 dual_sourcing 实例
    ds = dual_sourcing(c_r, c_e, h, b, l_r, l_e, demand, service_level)

    print('\n执行只用常规渠道的策略')
    single_source_result=ds.single_lost_sales(demand, S=None, inventory_level=0)
    print(f"只用单源策略平均总成本:{single_source_result['average_total_cost']}")


    lost_sales_result = ds.lost_sales(demand, S=None, inventory_level=0)
    print(lost_sales_result['order_record'])
    print("\n执行DDI双源策略...")
    ddi_result = ds.DDI_policy(demand, Se=None,D_2_constraint=True,inventory_level=0)
    print(f"DDI双源策略平均总成本: {ddi_result['average_total_cost']}")
    print(ds.cal_fill_rate(sample, ddi_result))
    # print(ddi_result['order_record_regular'])


    # print("\n执行SDI双源策略...")
    # sdi_result = ds.SDI_policy(demand, inventory_level=0)
    # print(f"SDI双源策略平均总成本: {sdi_result['average_total_cost']}")


    #调用TBS策略
    print('TBS')
    TBS_result=ds.TBS_policy(sample,demand,100,x_init=None,q_init=None)
    print(TBS_result['average_total_cost'])


    # # 调用DI策略
    di_cost = ds.DI_policy(demand, sample, x_init=None, q_init=None, inventory_level=0)
    print(f"DI策略平均总成本: {di_cost['average_total_cost']}")
    