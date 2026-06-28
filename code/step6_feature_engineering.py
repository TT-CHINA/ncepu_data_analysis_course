"""
step6_feature_engineering.py
特征工程 + 数据集划分 + 归一化
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ============================================================
# 1. 读取数据
# ============================================================
df = pd.read_csv('data/final_dataset.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"原始数据: {df.shape}")


# ============================================================
# 2. 构造滞后特征 (Lag Features) - 时序预测的核心
# ============================================================
# 用过去N天的用电量帮助预测今天
target = 'electricity_total'

for lag in [1, 2, 3, 7, 14, 30]:
    df[f'lag_{lag}'] = df[target].shift(lag)

print("✅ 滞后特征构造完成（lag_1, 2, 3, 7, 14, 30）")


# ============================================================
# 3. 滑动统计特征 (Rolling Statistics)
# ============================================================
# 过去7天、30天的均值、最大值、最小值、标准差
for window in [7, 30]:
    df[f'rolling_mean_{window}'] = df[target].shift(1).rolling(window).mean()
    df[f'rolling_std_{window}'] = df[target].shift(1).rolling(window).std()
    df[f'rolling_max_{window}'] = df[target].shift(1).rolling(window).max()
    df[f'rolling_min_{window}'] = df[target].shift(1).rolling(window).min()

print("✅ 滑动统计特征构造完成")


# ============================================================
# 4. 温度衍生特征（U型关系的关键）
# ============================================================
# 制冷度日 (Cooling Degree Day): 温度超过18°C的部分
# 采暖度日 (Heating Degree Day): 温度低于18°C的部分
df['CDD'] = (df['tavg'] - 18).clip(lower=0)
df['HDD'] = (18 - df['tavg']).clip(lower=0)

# 温度平方项（捕捉U型非线性）
df['tavg_sq'] = df['tavg'] ** 2

# 温度差（日内温差）
df['temp_range'] = df['tmax'] - df['tmin']

print("✅ 温度衍生特征构造完成（CDD, HDD, tavg_sq, temp_range）")


# ============================================================
# 5. 类别变量编码 (One-Hot)
# ============================================================
df = pd.get_dummies(df, columns=['season'], prefix='season')

# 转为0/1
season_cols = [c for c in df.columns if c.startswith('season_')]
df[season_cols] = df[season_cols].astype(int)

print(f"✅ 季节One-Hot编码完成")


# ============================================================
# 6. 删除滞后特征产生的NaN
# ============================================================
df_clean = df.dropna().reset_index(drop=True)
print(f"\n删除NaN后: {df_clean.shape} (删除了 {len(df) - len(df_clean)} 行)")


# ============================================================
# 7. 选择最终特征
# ============================================================
# 删除冗余/不能用的列
drop_cols = [
    'date',              # 时间索引，不参与建模
    'electricity_mean', 'electricity_max', 'electricity_min',  # 其他目标，建模时只用total
    'season_cn', 'weekday_cn', 'type', 'temp_bin', 'peak_valley_diff',  # EDA产生的辅助列
    'ma30',              # EDA画图时的滑动均值
]
drop_cols = [c for c in drop_cols if c in df_clean.columns]

feature_cols = [c for c in df_clean.columns if c not in drop_cols + [target]]
print(f"\n最终特征数: {len(feature_cols)}")
print("特征列表:")
for i, c in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {c}")


# ============================================================
# 8. 划分训练集 / 验证集 / 测试集（时序数据必须按时间划分！）
# ============================================================
n = len(df_clean)
train_end = int(n * 0.70)   # 70% 训练
val_end = int(n * 0.85)     # 15% 验证

train = df_clean.iloc[:train_end]
val = df_clean.iloc[train_end:val_end]
test = df_clean.iloc[val_end:]

print(f"\n数据集划分:")
print(f"  训练集: {len(train)} 条 ({train['date'].min().date()} ~ {train['date'].max().date()})")
print(f"  验证集: {len(val)} 条 ({val['date'].min().date()} ~ {val['date'].max().date()})")
print(f"  测试集: {len(test)} 条 ({test['date'].min().date()} ~ {test['date'].max().date()})")


# ============================================================
# 9. 归一化演示 (只在训练集上fit, 验证/测试集用同一个scaler transform)
# 说明：本步骤用于展示标准的特征工程与划分流程；差分预测建模所需的
#       归一化数据与 scaler 由 step7 统一生成（data/diff_data.npz、
#       models/scaler_*_diff.pkl），下游模型 step8/9/12 直接复用，
#       因此这里不再保存会与之冲突的 processed_data.npz / scaler_y.pkl。
# ============================================================
X_train = train[feature_cols].values
scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
print(f"\n归一化后训练特征形状: {X_train_s.shape}")


# ============================================================
# 10. 保存特征名与数据集划分日期（供 EDA / 报告引用）
# ============================================================
test[['date']].to_csv('data/test_dates.csv', index=False)
val[['date']].to_csv('data/val_dates.csv', index=False)

with open('data/feature_names.txt', 'w') as f:
    for c in feature_cols:
        f.write(c + '\n')

print("\n" + "=" * 60)
print("✅ 已保存:")
print("   data/feature_names.txt    - 特征名列表")
print("   data/test_dates.csv       - 测试集日期")
print("   data/val_dates.csv        - 验证集日期")
print("   （差分建模数据由 step7 统一生成）")
print("=" * 60)