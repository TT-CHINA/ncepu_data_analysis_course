"""
step3_process_electricity.py
处理Kaggle电力数据，聚合为日度
"""
import pandas as pd

# 读取原始数据
df_elec = pd.read_csv('data/AEP_hourly.csv')
print("原始数据：")
print(df_elec.head())
print(f"原始数据量: {len(df_elec)} 行")

# 转换日期
df_elec['Datetime'] = pd.to_datetime(df_elec['Datetime'])
df_elec['date'] = df_elec['Datetime'].dt.date

# 按日聚合（求和=日总用电量，求平均=日均小时负荷）
df_daily = df_elec.groupby('date').agg(
    electricity_total=('AEP_MW', 'sum'),    # 当日总用电（MWh）
    electricity_mean=('AEP_MW', 'mean'),    # 日均负荷（MW）
    electricity_max=('AEP_MW', 'max'),      # 日峰值
    electricity_min=('AEP_MW', 'min'),      # 日谷值
).reset_index()

df_daily['date'] = pd.to_datetime(df_daily['date'])

# 使用 AEP 数据完整可用区间，与气象数据交集对齐
# 起点取 2004-10-02（2004-10-01 仅 23 小时为不完整日，剔除以免低值离群）
df_daily = df_daily[(df_daily['date'] >= '2004-10-02') &
                     (df_daily['date'] <= '2018-08-02')]

print(f"\n聚合后日度数据: {len(df_daily)} 行")
print(df_daily.head())
print(f"\n统计：")
print(df_daily.describe())

df_daily.to_csv('data/electricity_daily.csv', index=False, encoding='utf-8-sig')
print("\n✅ 已保存到 data/electricity_daily.csv")