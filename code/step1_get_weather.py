"""
step1_get_weather.py
从 Open-Meteo 获取美国俄亥俄州 Columbus 2004-2018 年日度气象数据
（与 AEP 电力数据 2004-10 ~ 2018-08 的可用区间对齐）
"""
import requests
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    # Columbus, Ohio：AEP 服务区域内的代表城市
    "latitude": 39.9612,
    "longitude": -82.9988,
    "start_date": "2004-01-01",
    "end_date": "2018-12-31",
    "daily": ",".join([
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        "surface_pressure_mean",
        "relative_humidity_2m_mean",
        "sunshine_duration",
    ]),
    "timezone": "America/New_York"
}

print("正在从 Open-Meteo 获取 Columbus, Ohio 气象数据...")
response = requests.get(url, params=params, timeout=60)
print(f"HTTP 状态码: {response.status_code}")

if response.status_code != 200:
    print("请求失败，返回内容：")
    print(response.text)
    raise SystemExit

data = response.json()
df = pd.DataFrame(data['daily'])

# 重命名为简短字段
df.rename(columns={
    'time': 'date',
    'temperature_2m_mean': 'tavg',
    'temperature_2m_max': 'tmax',
    'temperature_2m_min': 'tmin',
    'precipitation_sum': 'prcp',
    'wind_speed_10m_max': 'wspd',
    'surface_pressure_mean': 'pres',
    'relative_humidity_2m_mean': 'rhum',
    'sunshine_duration': 'tsun',
}, inplace=True)

df['date'] = pd.to_datetime(df['date'])

print(f"\n✅ 共 {len(df)} 条记录")
print(f"字段: {df.columns.tolist()}")
print(f"\n前5行:")
print(df.head())
print(f"\n缺失情况:")
print(df.isnull().sum())
print(f"\n统计摘要:")
print(df.describe())

df.to_csv('data/columbus_weather.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ 已保存到 data/columbus_weather.csv")