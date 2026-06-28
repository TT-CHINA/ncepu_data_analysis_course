"""
step18_gru_residual.py
基于 BiGRU 在测试集上的预测结果（data/gru_results.csv）绘制残差诊断图，
生成图6-6「BiGRU 残差时序图与分布直方图」。
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('figures', exist_ok=True)

df = pd.read_csv('data/gru_results.csv', parse_dates=['date'])
res = df['truth'] - df['gru_pred']

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左：残差时序
axes[0].plot(df['date'], res, linewidth=0.8, color='purple', alpha=0.8)
axes[0].axhline(0, color='black', linewidth=1)
axes[0].axhline(res.mean(), color='red', linestyle='--', linewidth=1.2,
                label=f'均值={res.mean():.0f}')
axes[0].set_title('BiGRU 残差时序（真实值 − 预测值）', fontsize=13, fontweight='bold')
axes[0].set_xlabel('日期'); axes[0].set_ylabel('残差 (MWh)')
axes[0].legend(); axes[0].grid(alpha=0.3)

# 右：残差分布
axes[1].hist(res, bins=40, color='mediumpurple', edgecolor='black', alpha=0.8)
axes[1].axvline(res.mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值={res.mean():.0f}')
axes[1].axvline(0, color='black', linewidth=1)
axes[1].set_title('BiGRU 残差分布直方图', fontsize=13, fontweight='bold')
axes[1].set_xlabel('残差 (MWh)'); axes[1].set_ylabel('频数')
axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/25_gru_residual.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'✅ 图6-6 已生成 figures/25_gru_residual.png  残差均值={res.mean():.0f}  标准差={res.std():.0f}')
