import numpy as np
import scipy.stats

def sample_generation(distribution, sample_size, random_seed=None):
    """
    生成指定分布的随机样本
    
    参数:
    distribution: 元组 (分布名称, 参数元组)
    sample_size: 整数或元组 - 样本大小，如果是元组则生成多维数组
    random_seed: 随机种子
    
    返回:
    指定分布的随机样本
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    d = distribution[0]
    para = distribution[1]
    
    if d == 'norm':
        # 正态分布: (均值, 标准差)
        demand = scipy.stats.norm(loc=para[0], scale=para[1]).rvs(sample_size)
    
    elif d == 'beta':
        # 贝塔分布: (a, b)
        demand = scipy.stats.beta(a=para[0], b=para[1]).rvs(sample_size)
    
    elif d == 'binom':
        # 二项分布: (试验次数, 成功概率)
        demand = scipy.stats.binom(n=para[0], p=para[1]).rvs(sample_size)
    
    elif d == 'geom':
        # 几何分布: (成功概率)
        demand = scipy.stats.geom(p=para[0]).rvs(sample_size)
    
    elif d == 'expon':
        # 指数分布: (尺度参数)
        demand = scipy.stats.expon(scale=para[0]).rvs(sample_size)
    
    elif d == 'gamma':
        # 伽马分布: (形状参数, 尺度参数)
        demand = scipy.stats.gamma(a=para[0], scale=para[1]).rvs(sample_size)
    
    else:
        raise ValueError(f"不支持的分布类型: {d}")
    
    return demand
