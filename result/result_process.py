import pandas as pd
import pickle

results = []
with open("result/D1_result.pkl", "rb") as f:
    while True:
        try:
            results.append(pickle.load(f))
        except EOFError:
            break
print(len(results))
df = pd.DataFrame(results)
df.to_csv("result/D1_Result_raw.csv", index=False)

# 指定哪些列是参数（不参与平均）
param_cols = ["distribution", "c_e", "l_r", "l_e", "service_level", "b"]

# 对相同参数组合的结果取平均（排除 seed_id）
df_mean = (
    df.groupby(param_cols, as_index=False)
      .mean(numeric_only=True)  # 只对数值列求平均
)

# 保存为新文件
df_mean.to_csv("result/D1_Result_avg.csv", index=False)

print(f"✅ 平均值计算完成，共生成 {len(df_mean)} 条记录。文件已保存为 D1_Result_avg.csv")
