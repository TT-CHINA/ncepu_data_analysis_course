"""
step5_eda.py
数据探索分析与可视化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# 配置中文字体（Mac用'Heiti TC'/'Arial Unicode MS'，Windows用'SimHei'）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)

# 读取数据
df = pd.read_csv('data/final_dataset.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"数据加载完成：{df.shape}")


# ============================================================
# 图1：电力消费总体时序趋势
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['date'], df['electricity_total'], linewidth=0.8, color='steelblue', alpha=0.8)
# 添加30天滑动均值
df['ma30'] = df['electricity_total'].rolling(30).mean()
ax.plot(df['date'], df['ma30'], linewidth=2, color='red', label='30日滑动均值')
ax.set_title('电力消费日度时序趋势 (2015-2018)', fontsize=14, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('日总用电量 (MWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/01_time_series.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图1: 时序趋势")


# ============================================================
# 图2：月度&季节性模式（按年叠加）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左图：每月平均，按年份分组
monthly = df.groupby(['year', 'month'])['electricity_total'].mean().reset_index()
for year in sorted(monthly['year'].unique()):
    sub = monthly[monthly['year'] == year]
    axes[0].plot(sub['month'], sub['electricity_total'], marker='o', label=f'{year}年')
axes[0].set_title('月度电力消费模式（按年份）', fontsize=13, fontweight='bold')
axes[0].set_xlabel('月份')
axes[0].set_ylabel('月均日用电量 (MWh)')
axes[0].set_xticks(range(1, 13))
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右图：季节箱线图
season_order = ['spring', 'summer', 'autumn', 'winter']
season_cn = {'spring': '春', 'summer': '夏', 'autumn': '秋', 'winter': '冬'}
df['season_cn'] = df['season'].map(season_cn)
sns.boxplot(data=df, x='season_cn', y='electricity_total',
            order=['春', '夏', '秋', '冬'], ax=axes[1], palette='Set2')
axes[1].set_title('各季节电力消费分布', fontsize=13, fontweight='bold')
axes[1].set_xlabel('季节')
axes[1].set_ylabel('日总用电量 (MWh)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/02_seasonal.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图2: 季节性分析")


# ============================================================
# 图3：工作日 vs 周末 对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左图：星期分布箱线图
weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
df['weekday_cn'] = df['dayofweek'].apply(lambda x: weekday_names[x])
sns.boxplot(data=df, x='weekday_cn', y='electricity_total',
            order=weekday_names, ax=axes[0], palette='coolwarm')
axes[0].set_title('星期几对电力消费的影响', fontsize=13, fontweight='bold')
axes[0].set_xlabel('星期')
axes[0].set_ylabel('日总用电量 (MWh)')
axes[0].grid(alpha=0.3)

# 右图：工作日vs周末对比
df['type'] = df['is_weekend'].map({0: '工作日', 1: '周末'})
sns.violinplot(data=df, x='type', y='electricity_total', ax=axes[1], palette='pastel')
axes[1].set_title('工作日 vs 周末分布', fontsize=13, fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('日总用电量 (MWh)')
axes[1].grid(alpha=0.3)

# 计算差异
wd = df[df['is_weekend']==0]['electricity_total'].mean()
we = df[df['is_weekend']==1]['electricity_total'].mean()
print(f"   工作日平均: {wd:.0f} MWh | 周末平均: {we:.0f} MWh | 差值: {wd-we:.0f} ({(wd-we)/wd*100:.1f}%)")

plt.tight_layout()
plt.savefig('figures/03_weekday_weekend.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图3: 工作日vs周末")


# ============================================================
# 图4：节假日影响分析
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
holiday_df = df.groupby('is_holiday')['electricity_total'].agg(['mean', 'std']).reset_index()
holiday_df['label'] = holiday_df['is_holiday'].map({0: '非节假日', 1: '节假日'})

bars = ax.bar(holiday_df['label'], holiday_df['mean'],
              yerr=holiday_df['std'], capsize=10,
              color=['#5B9BD5', '#ED7D31'], alpha=0.8, edgecolor='black')
for bar, val in zip(bars, holiday_df['mean']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f'{val:.0f}', ha='center', fontsize=12, fontweight='bold')
ax.set_title('节假日 vs 非节假日 电力消费对比', fontsize=14, fontweight='bold')
ax.set_ylabel('日总用电量 (MWh)')
ax.grid(alpha=0.3, axis='y')

h0 = df[df['is_holiday']==0]['electricity_total'].mean()
h1 = df[df['is_holiday']==1]['electricity_total'].mean()
print(f"   非节假日: {h0:.0f} | 节假日: {h1:.0f} | 节假日降低 {(h0-h1)/h0*100:.1f}%")

plt.tight_layout()
plt.savefig('figures/04_holiday.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图4: 节假日影响")


# ============================================================
# 图5：温度 vs 电力消费（U型曲线，最重要的图）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左图：温度-电力散点图，按季节着色
sns.scatterplot(data=df, x='tavg', y='electricity_total', hue='season_cn',
                hue_order=['春', '夏', '秋', '冬'], alpha=0.5, ax=axes[0], palette='Set1')
# 拟合二次曲线
z = np.polyfit(df['tavg'], df['electricity_total'], 2)
p = np.poly1d(z)
x_smooth = np.linspace(df['tavg'].min(), df['tavg'].max(), 100)
axes[0].plot(x_smooth, p(x_smooth), 'k--', linewidth=2, label='二次拟合 (U型)')
axes[0].set_title('温度与电力消费的U型关系', fontsize=13, fontweight='bold')
axes[0].set_xlabel('日均温度 (°C)')
axes[0].set_ylabel('日总用电量 (MWh)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右图：温度分组的平均用电量
df['temp_bin'] = pd.cut(df['tavg'], bins=10)
temp_group = df.groupby('temp_bin', observed=True)['electricity_total'].mean().reset_index()
temp_group['temp_mid'] = temp_group['temp_bin'].apply(lambda x: x.mid)
axes[1].bar(range(len(temp_group)), temp_group['electricity_total'],
            color=plt.cm.coolwarm(np.linspace(0.1, 0.9, len(temp_group))),
            edgecolor='black')
axes[1].set_xticks(range(len(temp_group)))
axes[1].set_xticklabels([f'{x.mid:.0f}' for x in temp_group['temp_bin']], rotation=45)
axes[1].set_title('温度区间对应平均用电量', fontsize=13, fontweight='bold')
axes[1].set_xlabel('温度区间中值 (°C)')
axes[1].set_ylabel('平均日用电量 (MWh)')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/05_temp_electricity.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图5: 温度-电力U型关系")


# ============================================================
# 图6：相关性热图
# ============================================================
fig, ax = plt.subplots(figsize=(12, 9))
corr_cols = ['electricity_total', 'tavg', 'tmax', 'tmin', 'prcp', 'wspd',
             'pres', 'rhum', 'tsun', 'is_weekend', 'is_holiday', 'month', 'dayofweek']
corr_cols = [c for c in corr_cols if c in df.columns]
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('特征相关性热图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图6: 相关性热图")


# ============================================================
# 图7：时间序列分解（趋势+季节+残差）
# ============================================================
ts = df.set_index('date')['electricity_total']
decomp = seasonal_decompose(ts, model='additive', period=365)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
decomp.observed.plot(ax=axes[0], color='steelblue')
axes[0].set_title('原始序列', fontweight='bold'); axes[0].grid(alpha=0.3)
decomp.trend.plot(ax=axes[1], color='orange')
axes[1].set_title('趋势分量', fontweight='bold'); axes[1].grid(alpha=0.3)
decomp.seasonal.plot(ax=axes[2], color='green')
axes[2].set_title('季节分量', fontweight='bold'); axes[2].grid(alpha=0.3)
decomp.resid.plot(ax=axes[3], color='gray')
axes[3].set_title('残差分量', fontweight='bold'); axes[3].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/07_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图7: 时间序列分解")


# ============================================================
# 图8：月度热力图（年x月）
# ============================================================
pivot = df.pivot_table(index='month', columns='year',
                       values='electricity_total', aggfunc='mean')
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': '日均用电量 (MWh)'})
ax.set_title('年-月 电力消费热力图', fontsize=14, fontweight='bold')
ax.set_xlabel('年份')
ax.set_ylabel('月份')
plt.tight_layout()
plt.savefig('figures/08_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图8: 月度热力图")


# ============================================================
# 图9：电力消费分布与日内峰谷
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左图：日总用电量分布
axes[0].hist(df['electricity_total'], bins=40, color='steelblue',
             edgecolor='black', alpha=0.7)
axes[0].axvline(df['electricity_total'].mean(), color='red', linestyle='--',
                linewidth=2, label=f"均值: {df['electricity_total'].mean():.0f}")
axes[0].axvline(df['electricity_total'].median(), color='green', linestyle='--',
                linewidth=2, label=f"中位数: {df['electricity_total'].median():.0f}")
axes[0].set_title('日总用电量分布', fontsize=13, fontweight='bold')
axes[0].set_xlabel('日总用电量 (MWh)')
axes[0].set_ylabel('频数')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右图：峰谷差
df['peak_valley_diff'] = df['electricity_max'] - df['electricity_min']
monthly_diff = df.groupby('month')['peak_valley_diff'].mean()
axes[1].bar(monthly_diff.index, monthly_diff.values,
            color=plt.cm.viridis(np.linspace(0.2, 0.9, 12)), edgecolor='black')
axes[1].set_title('各月日内峰谷差均值', fontsize=13, fontweight='bold')
axes[1].set_xlabel('月份')
axes[1].set_ylabel('峰谷差 (MW)')
axes[1].set_xticks(range(1, 13))
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/09_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图9: 分布与峰谷差")


# ============================================================
# 关键统计结论输出
# ============================================================
print("\n" + "=" * 60)
print("📊 关键发现总结（可直接写入报告）")
print("=" * 60)
print(f"1. 数据范围: {df['date'].min().date()} 至 {df['date'].max().date()} (共{len(df)}天)")
print(f"2. 日均用电: {df['electricity_total'].mean():.0f} MWh")
print(f"3. 最高用电日: {df.loc[df['electricity_total'].idxmax(), 'date'].date()} ({df['electricity_total'].max():.0f} MWh)")
print(f"4. 最低用电日: {df.loc[df['electricity_total'].idxmin(), 'date'].date()} ({df['electricity_total'].min():.0f} MWh)")
print(f"5. 工作日 vs 周末: {wd:.0f} vs {we:.0f} (相差{(wd-we)/wd*100:.1f}%)")
print(f"6. 非节假日 vs 节假日: {h0:.0f} vs {h1:.0f} (节假日低{(h0-h1)/h0*100:.1f}%)")
print(f"7. 温度与电力相关性: {df['tavg'].corr(df['electricity_total']):.3f}")
print(f"8. 各季节均值:")
for s in ['spring', 'summer', 'autumn', 'winter']:
    mean = df[df['season']==s]['electricity_total'].mean()
    print(f"   {season_cn[s]}季: {mean:.0f} MWh")
print("\n✅ 所有图表已保存到 figures/ 文件夹")