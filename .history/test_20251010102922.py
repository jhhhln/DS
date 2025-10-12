import pandas as pd

# 读取原始数据
df = pd.read_csv("binom_experiment_results.csv")  # 替换为你的原始文件名

# 筛选出 seed_id == 9 的行
filtered_df = df[df["seed_id"] == 9]

# 保存为新的 CSV 文件
filtered_df.to_csv(".csv", index=False)

print(f"筛选完成，共保留 {len(filtered_df)} 条记录。文件已保存为 filtered_seed9.csv")
