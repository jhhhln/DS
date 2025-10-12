import pandas as pd

# # 读取原始数据
# df = pd.read_csv("geom_experiment_results.csv")  # 替换为你的原始文件名

# # 筛选出 seed_id == 9 的行
# filtered_df = df[df["seed_id"] == 9]

# # 保存为新的 CSV 文件
# filtered_df.to_csv("simp_geom_experiment_results.csv", index=False)

# print(f"筛选完成，共保留 {len(filtered_df)} 条记录。文件已保存为 filtered_seed9.csv")
import pandas as pd

# 读取原始数据
df = pd.read_csv("binom_experiment_results.csv") 

# 指定哪些列是参数（不参与平均）
param_cols = ["distribution", "c_e", "l_r", "l_e", "service_level", "b"]

# 对相同参数组合的结果取平均（排除 seed_id）
df_mean = (
    df.groupby(param_cols, as_index=False)
      .mean(numeric_only=True)  # 只对数值列求平均
)

# 保存为新文件
df_mean.to_csv("averaged_binom_results.csv", index=False)

print(f"✅ 平均值计算完成，共生成 {len(df_mean)} 条记录。文件已保存为 averaged_results.csv")
