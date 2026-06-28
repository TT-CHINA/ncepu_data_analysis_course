"""
step2_get_holiday.py
生成美国联邦节假日标记 + 时间特征
"""
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

# 生成日期序列（与气象数据对齐）
date_range = pd.date_range('2004-01-01', '2018-12-31', freq='D')
df_time = pd.DataFrame({'date': date_range})

# 1. 基础时间特征
df_time['year'] = df_time['date'].dt.year
df_time['month'] = df_time['date'].dt.month
df_time['day'] = df_time['date'].dt.day
df_time['dayofweek'] = df_time['date'].dt.dayofweek  # 0=周一, 6=周日
df_time['dayofyear'] = df_time['date'].dt.dayofyear
df_time['quarter'] = df_time['date'].dt.quarter
df_time['weekofyear'] = df_time['date'].dt.isocalendar().week

# 2. 季节
def get_season(month):
    if month in [3, 4, 5]: return 'spring'
    elif month in [6, 7, 8]: return 'summer'
    elif month in [9, 10, 11]: return 'autumn'
    else: return 'winter'
df_time['season'] = df_time['month'].apply(get_season)

# 3. 周末标记
df_time['is_weekend'] = (df_time['dayofweek'] >= 5).astype(int)

# 4. 美国联邦节假日（含观察日）
calendar = USFederalHolidayCalendar()
holidays = calendar.holidays(start=date_range.min(), end=date_range.max())

df_time['is_holiday'] = df_time['date'].isin(holidays).astype(int)
df_time['is_workday'] = ((df_time['is_weekend'] == 0) & (df_time['is_holiday'] == 0)).astype(int)

# 5. 周期性编码（深度学习常用技巧，避免12月和1月被认为很远）
df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
df_time['dayofweek_sin'] = np.sin(2 * np.pi * df_time['dayofweek'] / 7)
df_time['dayofweek_cos'] = np.cos(2 * np.pi * df_time['dayofweek'] / 7)

print(df_time.head(10))
print(f"\n共 {len(df_time)} 条")
print(f"节假日天数: {df_time['is_holiday'].sum()}")
print(f"周末天数: {df_time['is_weekend'].sum()}")

df_time.to_csv('data/time_features.csv', index=False, encoding='utf-8-sig')
print("\n✅ 已保存到 data/time_features.csv")