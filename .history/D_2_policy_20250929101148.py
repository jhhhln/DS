import numpy as np
from demand import sample_generation
 
class dual_sourcing:
    def __init__(self, c_r, c_e, h, b, l_r, l_e, demand, service_level):
        #�ɱ�����
        self.c_r = c_r
        self.c_e = c_e
        self.h   = h
        self.b   = b
        self.l_r = l_r
        self.l_e = l_e
        self.l=self.l_r-self.l_e
        #�����ķ���ˮƽalpha
        self.service_level = service_level

        # �������� [N, T] ������������һ��·���������ڣ���������·��������������demand�õ���cost���Ƚ�
        self.demand = demand
        #����������������ۼ���� ÿһ�д���k��D�Ŀ��ܵ�ȡֵ��֮���ÿһ�зֱ�ȡ��λ��
        self.cum_demand = self.demand.cumsum(axis=1)

        # S_1,��,S_{l_r+1}
        self.S_service_level = self.cum_demand_quantile()

        # �Ӽ���ʼ����
        self.q_init = np.zeros(self.l_e)

        # �����ʼ���� lost_sales��DDI
        self.x_init_DDI = np.diff(np.insert(self.S_service_level, 0, 0))[:self.l_r]

        # init x for SDI
        #ǰl_e+1��
        self.x_init_SDI=np.diff(np.insert(self.S_service_level, 0, 0))[:self.l_e+1]
        #��l_e+2����l_r��
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


#����S1,��S_(l_r+1)
    def cum_demand_quantile(self):
        #������������������ͣ�Ȼ���ÿһ�зֱ�ȡ��λ�������Ϊkȡ����(k+1)��D�����
        q_all = np.quantile(self.cum_demand,
                            q=self.service_level,
                            axis=0)  
        return q_all[:self.l_r + 1]

#����S_l����
    def Sl(self,c1,c2,h,b,l,alpha):
        quantile = (b*alpha + c2 - c1) / (h*(1-alpha) + b*alpha + c2 - c1)
        return np.quantile(self.cum_demand[:,l-1], quantile)

    #����lost-salesϵͳ����DDI���� �����Sһ��Ӧ����self.Sr
    def lost_sales(self, demand,S=None,inventory_level=0):
        S=self.Sr if S is None else S
        #һ��ģ���path����N
        iter_num = demand.shape[0]
        #ÿ��path�ĳ���T
        period_length = demand.shape[1]
        #ϣ����¼ÿһ�ڵ��ڳ����y,ÿһ�ڵĶ�����order�Լ�ÿһ�ڵĳɱ�
        #����ÿһ��·����������ͬ�ĳ�ʼ��������*N
        x_init = np.tile(self.x_init_DDI, (iter_num, 1))

        order_record = x_init.copy()
        inv_level_record = np.ones((iter_num, 1)) * inventory_level
        cost_per_period = np.zeros((iter_num, 0)) 
        y_level_record = np.zeros((iter_num, 1))

        for t in range(period_length):
            #�¶����Ĺ���ֻ��Ҫ����T-l_r�ڣ���Ϊ֮���µĶ������������ǿ��ǵ�T�ڵ���
            if order_record.shape[1] < period_length:
                
                # ���㶩�� ��һ�ڿ�ʼ֮���µĶ�������Ҫ����order_up_to������
                x_order = np.maximum(np.ones((iter_num, 1)) * S- order_record[:, t:t + self.l_r].sum(axis=1)[:, None]- inv_level_record[:, [t]],0)
                # ʹ�� hstack �����
                order_record = np.hstack((order_record, x_order))

            # ���㶩������֮��ı�����ͷ���(���+����)�ͱ��ڵ�����d
            y = inv_level_record[:, [t]] + order_record[:, [t]]
            d = demand[:, [t]]
            # ���㵱ǰ���ڵĳɱ�
            period_cost = (self.c_r * order_record[:, [t]]+ self.h * np.maximum(y - d, 0)+ self.b * np.maximum(d - y, 0))
            # ����ǰ���ڵĳɱ���ӵ��ɱ���¼��
            cost_per_period = np.hstack((cost_per_period, period_cost))

            # ������һ�ڵĿ��ˮƽ lost_salesϵͳ
            next_inv_level = np.maximum(y - d, 0)

            # ʹ�� hstack �����
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
            y_level_record = np.hstack((y_level_record, y))


        # ����ÿ���������У����ܳɱ�
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


#��DDI���Լ��Ͽ�������Լ�� ����ȷ���ܹ�����D_2��Լ��
    def DDI_policy(self,demand,Se=None,D_2_constraint=True,inventory_level=0):
        Se=self.Se if Se is None else Se
        lost_sales_record=self.lost_sales(demand=demand,S=None,inventory_level=0)

        iter_num = demand.shape[0]
        period_length = demand.shape[1]
        
        #����˫Դϵͳ�����������Ķ�������ÿһ��·����������ͬ�ĳ�ʼ������
        x_init = np.tile(self.x_init_DDI, (iter_num, 1))
        q_init= np.tile(self.q_init,(iter_num,1))
        #�����¼���������Ķ�����
        order_record_regular = x_init.copy()
        order_record_expe=q_init.copy()

        # ��ʼ����¼����
        inv_level_record = np.ones((iter_num, 1)) * inventory_level
        cost_per_period = np.zeros((iter_num, 0)) 
        y_level_record = np.zeros((iter_num, 1))

        for t in range(period_length):
            #������������˵���¶����Ĺ���ֻ��Ҫ����T-l_r�ڣ�֮���µĶ���������T���ڵ���
            if order_record_regular.shape[1] < period_length:
                # ȷ��������Խ��
                end_idx = min(t + self.l_r, order_record_regular.shape[1])
                #����t>=0֮���ÿһ���´�Ķ������Ǻ�lost_sales�еĽ����ͬ
                order_regular = lost_sales_record['order_record'][:, t+self.l_r][:, None]
                order_record_regular = np.hstack((order_record_regular, order_regular))
            #���ڿ�������˵���¶����Ĺ���ֻ��Ҫ����T-l_e�ڣ�֮���µĶ���Ҳ������T���ڵ���
            if order_record_expe.shape[1] < period_length:
                # ȷ��������Խ��
                end_idx = min(t + self.l_e, order_record_expe.shape[1])
                #���ݸ��µ����°���ԣ�ǰdelta l+1�ڵ��µ��ļӼ�������������Ϊ0
                if t<self.l+1:
                    order_e=np.zeros((iter_num, 1))
                    order_record_expe=np.hstack((order_record_expe,order_e))
                else:
                    #��ʼ��yΪ0 ����ȡ��
                    y_inv=lost_sales_record['y_level_record'][:, 1:]
                    net_inv = y_inv - demand
                    #����ÿһ��path��ÿһ�ڣ������Ӧʱ����backlogֵ
                    BACK = -np.minimum(net_inv, 0)

                    #����ÿ��ʱ���֮ǰ��������ǰʱ��㣩����� BACK ֵ
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

            # ���㵱ǰ���ڵĳɱ�
            period_cost = (
                self.c_r * order_record_regular[:, [t]]+self.c_e * order_record_expe[:, [t]]
                + self.h * np.maximum(y - d, 0)
                + self.b * np.maximum(d - y, 0)
            )

            
            # ����ǰ���ڵĳɱ���ӵ��ɱ���¼��
            cost_per_period = np.hstack((cost_per_period, period_cost))


            # ������һ�ڵĿ��ˮƽ
            next_inv_level = y - d

            
            # ʹ�� hstack �����
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
            y_level_record = np.hstack((y_level_record, y))


        # ����ÿ���������У����ܳɱ�
        total_cost_per_iteration = np.sum(cost_per_period, axis=1)
        average_total_cost = np.mean(total_cost_per_iteration)
        print(f"ƽ���ܳɱ�: {average_total_cost}")

        return {
            'order_record_regular': order_record_regular,
            'order_record_expe': order_record_expe,
            'inv_level_record': inv_level_record,
            'y_level_record': y_level_record,
            'cost_per_period': cost_per_period,
            'total_cost_per_iteration': total_cost_per_iteration,
            'average_total_cost': average_total_cost
        }


    def SDI_policy(self,demand,inventory_level=0):

        iter_num = demand.shape[0]
        period_length = demand.shape[1]

        #����˫Դϵͳ���������Ķ�����
        x_init=np.tile(self.x_init_SDI,(iter_num,1))
        q_init=np.tile(self.q_init,(iter_num,1))
        #��¼���������Ķ�����
        order_record_regular = x_init.copy()
        order_record_expe=q_init.copy()

        # ��ʼ����¼����
        inv_level_record = np.ones((iter_num, 1)) * inventory_level
        cost_per_period = np.zeros((iter_num, 0)) 
        y_level_record = np.zeros((iter_num, 1))
        
        for t in range(period_length):
            if order_record_expe.shape[1] < period_length:
                # ȷ��������Խ��
                IP_e=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e].sum(axis=1)[:,None]+order_record_regular[:,t:t+self.l_e+1].sum(axis=1)[:,None]
                order_e=np.maximum(np.ones(IP_e.shape) * self.Se - IP_e, 0)
                order_record_expe=np.hstack((order_record_expe,order_e))
            
            if order_record_regular.shape[1] < period_length:
                # ȷ��������Խ��
                end_idx = min(t + self.l_r, order_record_regular.shape[1])
                IP_r=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e+1].sum(axis=1)[:,None]+order_record_regular[:,t:t+self.l_r].sum(axis=1)[:,None]
                order_r=np.maximum(np.ones(IP_r.shape) * (self.Se+self.S_l) - IP_r, 0)
                order_record_regular=np.hstack((order_record_regular,order_r))

            y = inv_level_record[:, [t]] + order_record_regular[:, [t]]+order_record_expe[:,[t]]
            d=demand[:,[t]]

            # ���㵱ǰ���ڵĳɱ�
            period_cost = (
                self.c_r * order_record_regular[:, [t]]+self.c_e * order_record_expe[:, [t]]+ self.h * np.maximum(y - d, 0)+ self.b * np.maximum(d - y, 0))

            
            # ����ǰ���ڵĳɱ���ӵ��ɱ���¼��
            cost_per_period = np.hstack((cost_per_period, period_cost))


            # ������һ�ڵĿ��ˮƽ
            next_inv_level = y - d

            
            # ʹ�� hstack �����
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
            y_level_record = np.hstack((y_level_record, y))


        # ����ÿ���������У����ܳɱ�
        total_cost_per_iteration = np.sum(cost_per_period, axis=1)
        average_total_cost = np.mean(total_cost_per_iteration)
        print(f"ƽ���ܳɱ�: {average_total_cost}")

        return {
            'order_record_regular': order_record_regular,
            'order_record_expe': order_record_expe,
            'inv_level_record': inv_level_record,
            'y_level_record': y_level_record,
            'cost_per_period': cost_per_period,
            'total_cost_per_iteration': total_cost_per_iteration,
            'average_total_cost': average_total_cost
        }       

#�����Ǹ������������ͼӼ�������inventory position֮����ζ�����ÿ��·���ĳɱ�
    def cal_order_up_to(self,demand,S_e,S_r,x_init,q_init,inventory_level=0):

        iter_num = demand.shape[0]
        period_length = demand.shape[1]

        #����˫Դϵͳ���������Ķ�����
        x_init=np.tile(x_init,(iter_num,1))
        q_init=np.tile(q_init,(iter_num,1))
        #��¼���������Ķ�����
        order_record_regular = x_init.copy()
        order_record_expe=q_init.copy()

        # ��ʼ����¼����
        inv_level_record = np.ones((iter_num, 1)) * inventory_level
        cost_per_period = np.zeros((iter_num, 0)) 
        y_level_record = np.zeros((iter_num, 1))
        overshoot_record=np.zeros((iter_num, 1))

        for t in range(period_length):
            if order_record_expe.shape[1] < period_length:
                # ȷ��������Խ��
                end_idx = min(t + self.l_e, order_record_expe.shape[1])
                IP_e=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e].sum(axis=1)[:,None]+order_record_regular[:,t:t+self.l_e+1].sum(axis=1)[:,None]
                order_e=np.maximum(np.ones(IP_e.shape) * S_e - IP_e, 0)
                order_record_expe=np.hstack((order_record_expe,order_e))
                overshoot=np.maximum(IP_e-np.ones(IP_e.shape) * S_e, 0)
                overshoot_record=np.hstack((overshoot_record,overshoot))
            
            if order_record_regular.shape[1] < period_length:
                # ȷ��������Խ��
                end_idx = min(t + self.l_r, order_record_regular.shape[1])
                IP_r=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e+1].sum(axis=1)[:,None]+order_record_regular[:,t:t+self.l_r].sum(axis=1)[:,None]
                order_r=np.maximum(np.ones(IP_r.shape) * S_r - IP_r, 0)
                order_record_regular=np.hstack((order_record_regular,order_r))

            y = inv_level_record[:, [t]] + order_record_regular[:, [t]]+order_record_expe[:,[t]]
            d=demand[:,[t]]

            # ���㵱ǰ���ڵĳɱ�
            period_cost = (
                self.c_r * order_record_regular[:, [t]]+self.c_e * order_record_expe[:, [t]]+ self.h * np.maximum(y - d, 0)+ self.b * np.maximum(d - y, 0))

            
            # ����ǰ���ڵĳɱ���ӵ��ɱ���¼��
            cost_per_period = np.hstack((cost_per_period, period_cost))


            # ������һ�ڵĿ��ˮƽ
            next_inv_level = y - d

            
            # ʹ�� hstack �����
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
            y_level_record = np.hstack((y_level_record, y))


        # ����ÿ���������У����ܳɱ�
        total_cost_per_iteration = np.sum(cost_per_period, axis=1)
        average_total_cost = np.mean(total_cost_per_iteration)

        return {
            'order_record_regular': order_record_regular,
            'order_record_expe': order_record_expe,
            'inv_level_record': inv_level_record,
            'y_level_record': y_level_record,
            'cost_per_period': cost_per_period,
            'total_cost_per_iteration': total_cost_per_iteration,
            'average_total_cost': average_total_cost,
            'overshoot_record':overshoot_record
        }       


    def DI_policy(self,demand,sample,x_init=None,q_init=None,inventory_level=0):
        #���ҵ�delta����̬�ֲ���Ȼ������������ŵ���ϣ�S_e,S_e+delta),��������������ųɱ�
        #������sample path�ҵ���̬�ֲ�
        x_init=self.x_init_SDI if x_init is None else x_init
        q_init=self.q_init if q_init is None else q_init

        delta_range = np.arange(0, self.Se /self.num_search_range*(self.num_search_range+1), self.Se/self.num_search_range)

        DI_cost_record =[]
        for delta in delta_range:
            record=self.cal_order_up_to(sample,self.Se,delta+self.Se,x_init,q_init,inventory_level=0)
           
            cost=record['average_total_cost']
            DI_cost_record.append(cost)
        min_cost_idx = np.argmin(DI_cost_record)
        optimal_delta = delta_range[min_cost_idx]


        record_of_demand=self.cal_order_up_to(demand,self.Se,self.Se+optimal_delta,x_init,q_init,inventory_level=0)
        return record_of_demand['average_total_cost']
#TBS�����мӼ�������order_up_to,����������r
    def cal_order_up_to_with_r(self,demand,S_e,r,x_init,q_init,inventory_level=0):

        iter_num = demand.shape[0]
        period_length = demand.shape[1]

        #����˫Դϵͳ���������Ķ�����
        x_init=np.tile(x_init,(iter_num,1))
        q_init=np.tile(q_init,(iter_num,1))
        #��¼���������Ķ�����
        order_record_regular = x_init.copy()
        order_record_expe=q_init.copy()

        # ��ʼ����¼����
        inv_level_record = np.ones((iter_num, 1)) * inventory_level
        cost_per_period = np.zeros((iter_num, 0)) 
        y_level_record = np.zeros((iter_num, 1))

        for t in range(period_length):
            if order_record_expe.shape[1] < period_length:
                # ȷ��������Խ��
                end_idx = min(t + self.l_e, order_record_expe.shape[1])
                IP_e=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e].sum(axis=1)[:,None]+order_record_regular[:,t:t+self.l_e+1].sum(axis=1)[:,None]
                order_e=np.maximum(np.ones(IP_e.shape) * S_e - IP_e, 0)
                order_record_expe=np.hstack((order_record_expe,order_e))
            
            if order_record_regular.shape[1] < period_length:
                # ȷ��������Խ��
                end_idx = min(t + self.l_r, order_record_regular.shape[1])
                IP_r=inv_level_record[:,[t]]+order_record_expe[:,t:t+self.l_e+1].sum(axis=1)[:,None]+order_record_regular[:,t:t+self.l_r].sum(axis=1)[:,None]
                order_r=np.ones(IP_r.shape)*r
                order_record_regular=np.hstack((order_record_regular,order_r))

            y = inv_level_record[:, [t]] + order_record_regular[:, [t]]+order_record_expe[:,[t]]
            d=demand[:,[t]]

            # ���㵱ǰ���ڵĳɱ�
            period_cost = (
                self.c_r * order_record_regular[:, [t]]+self.c_e * order_record_expe[:, [t]]
                + self.h * np.maximum(y - d, 0)
                + self.b * np.maximum(d - y, 0)
            )

            
            # ����ǰ���ڵĳɱ���ӵ��ɱ���¼��
            cost_per_period = np.hstack((cost_per_period, period_cost))


            # ������һ�ڵĿ��ˮƽ
            next_inv_level = y - d

            
            # ʹ�� hstack �����
            inv_level_record = np.hstack((inv_level_record, next_inv_level))
            y_level_record = np.hstack((y_level_record, y))


        # ����ÿ���������У����ܳɱ�
        total_cost_per_iteration = np.sum(cost_per_period, axis=1)
        average_total_cost = np.mean(total_cost_per_iteration)

        return {
            'order_record_regular': order_record_regular,
            'order_record_expe': order_record_expe,
            'inv_level_record': inv_level_record,
            'y_level_record': y_level_record,
            'cost_per_period': cost_per_period,
            'total_cost_per_iteration': total_cost_per_iteration,
            'average_total_cost': average_total_cost
        }       

    def TBS_policy(self,sample,demand,mean,x_init=None,q_init=None):
        x_init=self.x_init_SDI if x_init is None else x_init
        q_init=self.q_init if q_init is None else q_init
        r_range = np.linspace(0,mean,self.num_search_range+1)
        TBS_cost_record =[]
        for r in r_range:
            record=self.cal_order_up_to_with_r(sample,self.Se,r,x_init,q_init,inventory_level=0)
            cost=record['average_total_cost']
            TBS_cost_record.append(cost)
        min_cost_idx = np.argmin(TBS_cost_record)
        best_r=r_range[min_cost_idx]
        optimal_record=self.cal_order_up_to_with_r(demand,self.Se,best_r,x_init,q_init,inventory_level=0)
        return optimal_record    
            
            
            
            
if __name__ == "__main__":
    # ���ò���
    c_r = 0    # ���涩���ɱ�
    c_e = 10   # �Ӽ������ɱ�
    h = 1      # �����гɱ�

    l_r = 3  # ���涩����ǰ��
    l_e = 1    # �Ӽ�������ǰ��
    b = c_e+h*(l_r+1)    # ȱ���ɱ�
    T = 90   # ʱ��������
    N = 500  # ģ��·������
    service_level = 0.9  # ����ˮƽ
    
    # ������������ - ʹ����̬�ֲ�
    distribution = ('norm', (100, 10))  # ��ֵΪ10����׼��Ϊ10����̬�ֲ�
    demand = sample_generation(distribution, (N, T), random_seed=36)
    sample= sample_generation(distribution, (N, 500), random_seed=42)
    
    # ���� dual_sourcing ʵ��
    ds = dual_sourcing(c_r, c_e, h, b, l_r, l_e, demand, service_level)
    

    print("\nִ��DDI˫Դ����...")
    ddi_result = ds.DDI_policy(demand, Se=None,D_2_constraint=True,inventory_level=0)
    print(f"DDI˫Դ����ƽ���ܳɱ�: {ddi_result['average_total_cost']}")


    print("\nִ��SDI˫Դ����...")
    sdi_result = ds.SDI_policy(demand, inventory_level=0)
    print(f"SDI˫Դ����ƽ���ܳɱ�: {sdi_result['average_total_cost']}")


    #����TBS����
    print('TBS')
    TBS_result=ds.TBS_policy(sample,demand,100,x_init=None,q_init=None)
    print(TBS_result['average_total_cost'])


    # # ����DI����
    di_cost = ds.DI_policy(demand, sample, x_init=None, q_init=None, inventory_level=0)
    print(f"DI����ƽ���ܳɱ�: {di_cost}")
    