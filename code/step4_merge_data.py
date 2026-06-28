"""
step4_merge_data.py
合并气象、时间特征、电力数据
"""
import pandas as pd

# 读取三份数据
df_weather = pd.read_csv('data/columbus_weather.csv', parse_dates=['date'])
df_time = pd.read_csv('data/time_features.csv', parse_dates=['date'])
df_elec = pd.read_csv('data/electricity_daily.csv', parse_dates=['date'])

# 按日期依次合并
df = df_elec.merge(df_weather, on='date', how='left')
df = df.merge(df_time, on='date', how='left')

print(f"合并后数据量: {len(df)} 行, {len(df.columns)} 列")
print(f"\n所有字段:")
print(df.columns.tolist())
print(f"\n缺失值情况:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 处理气象缺失值（用前向填充+后向填充）
weather_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
for col in weather_cols:
    if col in df.columns:
        df[col] = df[col].ffill().bfill()

# 删除缺失过多的列（如snow, tsun可能缺失严重）
threshold = 0.5
df = df.dropna(axis=1, thresh=int(len(df) * threshold))

print(f"\n处理后字段:")
print(df.columns.tolist())

# 保存最终数据集
df.to_csv('data/final_dataset.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ 最终数据集已保存到 data/final_dataset.csv")
print(f"   共 {len(df)} 行, {len(df.columns)} 列")
print(f"\n前5行预览:")
print(df.head())