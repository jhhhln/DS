import itertools
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from D_1_policy import dual_sourcing
from demand import sample_generation
import pickle
import multiprocessing
import gc
import time

# 固定参数
c_r = 0
h = 1
T = 90
N = 1000
N_1 = 100

# 参数范围
c_e_list = [1, 2]
lt_pairs = [(2,1), (3, 1), (5, 2), (8,3), (10, 3)]
service_level_list = [0.95, 0.975, 0.99]
distributions = [("norm", (100, 10)), ("geom", (0.4,)), ("binom", (100, 0.5))]
# distributions = [("geom", (0.4,))]

def run_simulation(args):
    try:
        dist, c_e, lt_pair, service_level, seed_id = args
        l_r, l_e = lt_pair
        b = c_e + h * (l_r + 1)

        if b / (b + h) >= service_level:
            return None

        random_seed1 = np.random.randint(10000)
        random_seed2 = np.random.randint(10000)

        demand = sample_generation(dist, (N, T), random_seed=random_seed1)
        demand[demand < 0] = 0
        sample = sample_generation(dist, (N_1, 1000), random_seed=random_seed2)
        sample[sample < 0] = 0

        ds = dual_sourcing(c_r, c_e, h, b, l_r, l_e, 90, sample, service_level)

        single_result = ds.single_source(demand)
        ddi_result = ds.DDI_policy(demand)
        tbs_result = ds.constrained_TBS_policy(sample, demand)
        benchmark_tbs_result = ds.cost_driven_TBS_policy(sample, demand)
        benchmark_di_result = ds.benchmark_DI_policy(demand, sample)
        di_result = ds.constrained_DI_policy(demand, sample)

        return {
            "distribution": dist[0],
            "c_e": c_e,
            "l_r": l_r,
            "l_e": l_e,
            "service_level": service_level,
            "seed_id": seed_id,
            "b": b,
            "single_cost": single_result['average_total_cost'],
            "DDI_cost": ddi_result['average_total_cost'],
            "TBS_cost": tbs_result['average_total_cost'],
            "benchmark_TBS_result": benchmark_tbs_result['average_total_cost'],
            "benchmark_DI_result": benchmark_di_result['average_total_cost'],
            "DI_result": di_result['average_total_cost'],
            "random_seed1": random_seed1,
            "random_seed2": random_seed2,
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "args": args}


# --------------------------
# 主程序（入口必须写在这里）
# --------------------------
if __name__ == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # 参数组合
    all_tasks = [
        (dist, c_e, lt_pair, service_level, seed_id)
        for dist, c_e, lt_pair, service_level in itertools.product(distributions, c_e_list, lt_pairs, service_level_list)
        for seed_id in range(10)
    ]

    # 自动选择 CPU 核数（保留一个核心用于系统）
    # max_workers = max(1, int(multiprocessing.cpu_count())-1)
    max_workers = 4
    print(f"🧠 使用 {max_workers} 个进程并行计算...")
    # 输出文件
    output_file = "./result/D1_result.pkl"
    if os.path.exists(output_file):
        os.remove(output_file)

    # -----------------------------
    # 🚀 主循环：异步执行 + 流式写入
    # -----------------------------
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_simulation, args) for args in all_tasks]
        for f in tqdm(as_completed(futures), total=len(futures)):
            try:
                res = f.result()
                if res:
                    # ✳️ 分批写入磁盘，防止 results 过大导致变慢
                    with open(output_file, "ab") as fout:
                        pickle.dump(res, fout)
            except Exception as e:
                print(f"⚠️ 某个任务出错: {e}")
            finally:
                # 主动清理内存，防止堆积导致程序变慢
                gc.collect()

    print(f"\n✅ 所有模拟完成，结果已分批保存为 {output_file}")