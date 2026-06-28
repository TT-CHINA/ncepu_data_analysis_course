"""
step10_compare_analyze.py
三模型综合对比 + 影响因素分析
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)

# ============================================================
# 1. 加载三个模型的预测结果
# ============================================================
lstm_df = pd.read_csv('data/lstm_results.csv', parse_dates=['date'])
gru_df = pd.read_csv('data/gru_results.csv', parse_dates=['date'])
cnn_df = pd.read_csv('data/cnn_lstm_results.csv', parse_dates=['date'])

results = lstm_df[['date', 'truth', 'lstm_pred']].merge(
    gru_df[['date', 'gru_pred']], on='date').merge(
    cnn_df[['date', 'cnn_lstm_pred']], on='date')
print(f"对比数据: {results.shape}")

# 加载指标
lstm_m = pd.read_csv('data/lstm_metrics.csv')
gru_m = pd.read_csv('data/gru_metrics.csv')
cnn_m = pd.read_csv('data/cnn_lstm_metrics.csv')
metrics_df = pd.concat([lstm_m, gru_m, cnn_m], ignore_index=True)
print("\n模型对比指标:")
print(metrics_df[['model', 'RMSE', 'MAE', 'MAPE', 'R2']].to_string(index=False))


# ============================================================
# 图1: 三模型预测曲线对比
# ============================================================
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(results['date'], results['truth'], label='真实值', linewidth=2.5, color='black', alpha=0.85)
ax.plot(results['date'], results['lstm_pred'], label=f'LSTM (MAPE={lstm_m["MAPE"].iloc[0]:.2f}%)',
        linewidth=1.5, linestyle='--', color='#E74C3C', alpha=0.85)
ax.plot(results['date'], results['gru_pred'], label=f'GRU (MAPE={gru_m["MAPE"].iloc[0]:.2f}%)',
        linewidth=1.5, linestyle='--', color='#27AE60', alpha=0.85)
ax.plot(results['date'], results['cnn_lstm_pred'], label=f'CNN-LSTM (MAPE={cnn_m["MAPE"].iloc[0]:.2f}%)',
        linewidth=1.5, linestyle='--', color='#8E44AD', alpha=0.85)
ax.set_title('三种深度学习模型预测对比', fontsize=14, fontweight='bold')
ax.set_xlabel('日期'); ax.set_ylabel('日总用电量 (MWh)')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/15_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ 图15: 三模型预测对比")


# ============================================================
# 图2: 指标对比柱状图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
colors = ['#E74C3C', '#27AE60', '#8E44AD']
model_names = ['LSTM', 'GRU', 'CNN-LSTM']

# MAPE
ax = axes[0, 0]
bars = ax.bar(model_names, metrics_df['MAPE'], color=colors, edgecolor='black', alpha=0.85)
for b, v in zip(bars, metrics_df['MAPE']):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{v:.2f}%',
            ha='center', fontweight='bold', fontsize=11)
ax.set_title('MAPE 对比 (越低越好)', fontsize=12, fontweight='bold')
ax.set_ylabel('MAPE (%)'); ax.grid(alpha=0.3, axis='y')

# RMSE
ax = axes[0, 1]
bars = ax.bar(model_names, metrics_df['RMSE'], color=colors, edgecolor='black', alpha=0.85)
for b, v in zip(bars, metrics_df['RMSE']):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+200, f'{v:.0f}',
            ha='center', fontweight='bold', fontsize=11)
ax.set_title('RMSE 对比 (越低越好)', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE (MWh)'); ax.grid(alpha=0.3, axis='y')

# R²
ax = axes[1, 0]
bars = ax.bar(model_names, metrics_df['R2'], color=colors, edgecolor='black', alpha=0.85)
for b, v in zip(bars, metrics_df['R2']):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.4f}',
            ha='center', fontweight='bold', fontsize=11)
ax.set_title('R² 对比 (越高越好)', fontsize=12, fontweight='bold')
ax.set_ylabel('R²'); ax.grid(alpha=0.3, axis='y')

# 训练时间
ax = axes[1, 1]
times = [lstm_m['training_time'].iloc[0], gru_m['training_time'].iloc[0], cnn_m['training_time'].iloc[0]]
bars = ax.bar(model_names, times, color=colors, edgecolor='black', alpha=0.85)
for b, v in zip(bars, times):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{v:.1f}s',
            ha='center', fontweight='bold', fontsize=11)
ax.set_title('训练时间对比', fontsize=12, fontweight='bold')
ax.set_ylabel('训练时间 (秒)'); ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/16_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图16: 指标对比柱状图")


# ============================================================
# 图3: 误差箱线图 (各模型预测误差分布)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

errors_df = pd.DataFrame({
    'LSTM': results['truth'] - results['lstm_pred'],
    'GRU': results['truth'] - results['gru_pred'],
    'CNN-LSTM': results['truth'] - results['cnn_lstm_pred']
})

sns.boxplot(data=errors_df, ax=axes[0], palette=['#E74C3C', '#27AE60', '#8E44AD'])
axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0].set_title('预测误差分布', fontsize=12, fontweight='bold')
axes[0].set_ylabel('误差 (MWh)')
axes[0].grid(alpha=0.3, axis='y')

# 绝对百分比误差
abs_pct_df = pd.DataFrame({
    'LSTM': np.abs((results['truth'] - results['lstm_pred']) / results['truth']) * 100,
    'GRU': np.abs((results['truth'] - results['gru_pred']) / results['truth']) * 100,
    'CNN-LSTM': np.abs((results['truth'] - results['cnn_lstm_pred']) / results['truth']) * 100
})
sns.boxplot(data=abs_pct_df, ax=axes[1], palette=['#E74C3C', '#27AE60', '#8E44AD'])
axes[1].set_title('绝对百分比误差分布', fontsize=12, fontweight='bold')
axes[1].set_ylabel('|百分比误差| (%)')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/17_error_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图17: 误差分布对比")


# ============================================================
# 图4: 散点图三联 (预测 vs 真实)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, col, name, color in zip(axes,
    ['lstm_pred', 'gru_pred', 'cnn_lstm_pred'],
    ['LSTM', 'GRU', 'CNN-LSTM'],
    ['#E74C3C', '#27AE60', '#8E44AD']):
    
    ax.scatter(results['truth'], results[col], alpha=0.5, color=color, edgecolor='black', s=40)
    min_v = min(results['truth'].min(), results[col].min())
    max_v = max(results['truth'].max(), results[col].max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1.5, label='理想')
    
    r2 = 1 - np.sum((results['truth']-results[col])**2) / np.sum((results['truth']-results['truth'].mean())**2)
    ax.set_title(f'{name} (R²={r2:.3f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('真实值 (MWh)')
    ax.set_ylabel('预测值 (MWh)')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/18_scatter_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图18: 散点对比三联图")


# ============================================================
# 图5: 影响因素分析 - 用随机森林做特征重要性
# (深度学习的注意力机制可视化太复杂,用RF做特征重要性更直观)
# ============================================================
print("\n开始特征重要性分析...")

data = np.load('data/diff_data.npz')
X_train = data['X_train']
y_train_diff = data['y_train_diff'].flatten()

# 特征名直接来自公共模块，保证与训练时口径一致
_, diff_features = utils.build_diff_frame()
print(f"特征数: {len(diff_features)}")

# 训练随机森林
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_diff)

# 特征重要性
imp_df = pd.DataFrame({
    'feature': diff_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 重要特征:")
print(imp_df.head(15).to_string(index=False))

# 画特征重要性图
fig, ax = plt.subplots(figsize=(12, 8))
top_n = 20
top_features = imp_df.head(top_n).iloc[::-1]
bars = ax.barh(top_features['feature'], top_features['importance'],
               color=plt.cm.viridis(np.linspace(0.2, 0.9, top_n)),
               edgecolor='black')
ax.set_xlabel('特征重要性', fontsize=12)
ax.set_title(f'特征重要性排名 Top {top_n} (基于随机森林)', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('figures/19_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图19: 特征重要性")


# ============================================================
# 图6: 影响因素分类汇总
# ============================================================
# 把特征按类型分组,看哪类因素更重要
def categorize(feat):
    if feat.startswith('lag_') or feat.startswith('rolling_') or feat.startswith('diff_lag_'):
        return '历史用电特征'
    if feat in ['tavg', 'tmax', 'tmin', 'prcp', 'wspd', 'pres', 'rhum', 'tsun',
                'CDD', 'HDD', 'tavg_sq', 'temp_range']:
        return '气象因素'
    if feat in ['is_weekend', 'is_holiday', 'is_workday']:
        return '日历特征'
    if feat.startswith('season_'):
        return '季节'
    if feat in ['month', 'dayofweek', 'quarter', 'weekofyear', 'dayofyear',
                'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos']:
        return '周期编码'
    if feat in ['year', 'day']:
        return '时间标识'
    return '其他'

imp_df['category'] = imp_df['feature'].apply(categorize)
cat_imp = imp_df.groupby('category')['importance'].sum().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors_cat = plt.cm.Set2(np.linspace(0, 1, len(cat_imp)))
bars = ax.barh(cat_imp.index, cat_imp.values, color=colors_cat, edgecolor='black')
for b, v in zip(bars, cat_imp.values):
    ax.text(b.get_width()+0.005, b.get_y()+b.get_height()/2, f'{v:.3f} ({v*100:.1f}%)',
            va='center', fontsize=10)
ax.set_xlabel('累计重要性', fontsize=12)
ax.set_title('影响电力消费的因素类别贡献度', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('figures/20_factor_categories.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图20: 影响因素分类")


# ============================================================
# 综合结论汇总
# ============================================================
print("\n" + "=" * 60)
print("📊 综合分析结论")
print("=" * 60)

best_idx = metrics_df['MAPE'].idxmin()
print(f"\n【模型对比】")
print(f"最佳模型: {metrics_df.iloc[best_idx]['model']}")
print(f"  - MAPE: {metrics_df.iloc[best_idx]['MAPE']:.2f}%")
print(f"  - R²:   {metrics_df.iloc[best_idx]['R2']:.4f}")

print(f"\n【三模型相对差异】")
print(metrics_df[['model', 'MAPE', 'R2']].to_string(index=False))

print(f"\n【影响因素排名】")
print(f"  类别贡献度:")
for cat, imp in cat_imp.sort_values(ascending=False).items():
    print(f"    {cat:12s}: {imp*100:5.1f}%")

print(f"\n【单一特征 Top 5】")
for _, row in imp_df.head(5).iterrows():
    print(f"    {row['feature']:20s}: {row['importance']:.4f}")

# 保存最终结果
metrics_df.to_csv('data/final_comparison.csv', index=False)
imp_df.to_csv('data/feature_importance.csv', index=False)

print("\n✅ 全部完成！")
print("📂 生成的图表:")
print("   figures/15_model_comparison.png   三模型预测曲线对比")
print("   figures/16_metrics_comparison.png 指标柱状图对比")
print("   figures/17_error_distribution.png 误差分布对比")
print("   figures/18_scatter_comparison.png 散点图三联")
print("   figures/19_feature_importance.png 特征重要性 Top 20")
print("   figures/20_factor_categories.png  影响因素分类")