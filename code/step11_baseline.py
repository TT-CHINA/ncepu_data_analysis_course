"""
step11_baseline.py
基线模型对比：持续法(Persistence)、周季节朴素法、线性回归。
在与深度模型完全相同的测试区间（164天）上评估，用于衬托深度学习的增益。
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)
utils.set_seed()

data = utils.prepare_diff_data()
SEQ_LEN = utils.SEQ_LEN
TARGET = utils.TARGET

# 与深度模型一致：测试区间为 test 切片 iloc[SEQ_LEN:]，共 164 天
test = data['test'].reset_index(drop=True)
truth = data['truth_test'][SEQ_LEN:]
prev = data['prev_test'][SEQ_LEN:]              # 前一天真实值 = 持续法预测
dates = test['date'].iloc[SEQ_LEN:].reset_index(drop=True)

# 1. 持续法（昨天=今天）
persistence_pred = prev

# 2. 周季节朴素法（上周同一天）
weekly_pred = test[TARGET].shift(7).values[SEQ_LEN:]

# 3. 线性回归（同样预测差分后还原，与深度模型流程一致）
lr = LinearRegression()
lr.fit(data['X_train_s'], data['y_train_s'].ravel())
lr_diff_scaled = lr.predict(data['X_test_s'][SEQ_LEN:]).reshape(-1, 1)
lr_diff = data['scaler_y'].inverse_transform(lr_diff_scaled).flatten()
lr_pred = prev + lr_diff

baselines = {
    'Persistence (昨天=今天)': persistence_pred,
    'Seasonal Naive (上周同日)': weekly_pred,
    'Linear Regression (差分)': lr_pred,
}

rows = []
for name, pred in baselines.items():
    m = utils.compute_metrics(pred, truth)
    rows.append({'model': name, **m})
    print(f"{name:28s} | RMSE {m['RMSE']:9.1f} | MAE {m['MAE']:9.1f} | "
          f"MAPE {m['MAPE']:.2f}% | R² {m['R2']:.4f}")

baseline_df = pd.DataFrame(rows)
baseline_df.to_csv('data/baseline_metrics.csv', index=False)

# 与深度学习最优模型(CNN-LSTM)对照
dl = pd.read_csv('data/final_comparison.csv')
best_dl = dl.loc[dl['MAPE'].idxmin()]
print(f"\n深度学习最优: {best_dl['model']} | MAPE {best_dl['MAPE']:.2f}% | R² {best_dl['R2']:.4f}")

# ============================================================
# 图：基线 vs 深度学习 MAPE 对比
# ============================================================
compare = baseline_df[['model', 'MAPE']].copy()
compare = pd.concat([compare,
                     pd.DataFrame([{'model': f"{best_dl['model']} (本研究最优)",
                                    'MAPE': best_dl['MAPE']}])], ignore_index=True)

fig, ax = plt.subplots(figsize=(11, 6))
colors = ['#95a5a6', '#95a5a6', '#f39c12', '#27AE60']
bars = ax.barh(compare['model'], compare['MAPE'], color=colors, edgecolor='black', alpha=0.9)
for b, v in zip(bars, compare['MAPE']):
    ax.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2,
            f'{v:.2f}%', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('MAPE (%) — 越低越好', fontsize=12)
ax.set_title('基线模型 vs 深度学习模型 (相同测试区间)', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='x')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('figures/21_baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图21: 基线对比已保存")
print("✅ data/baseline_metrics.csv 已保存")
