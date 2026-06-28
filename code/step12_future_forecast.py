"""
step12_future_forecast.py
递归多步预测：用最优模型(CNN-LSTM)对最后 HORIZON 天做"滚动多步"预测。

与单步回测的区别：
- 单步：每天用真实的前一天值还原差分（教师强制），评估模型逐日预测能力。
- 多步：把模型自己的预测值回灌作为历史，递归生成未来曲线，评估真实部署场景。
气象与日历属于已知外生变量（部署时可由天气预报/日历提供），不回灌。
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)
utils.set_seed()
device = torch.device('cpu')

TARGET = utils.TARGET
SEQ_LEN = utils.SEQ_LEN
HORIZON = 30

# 特征列顺序（与训练一致）
_, feature_cols = utils.build_diff_frame()
scaler_X = joblib.load('models/scaler_X_diff.pkl')
scaler_y = joblib.load('models/scaler_y_diff.pkl')

BEST_NAME = 'LSTM'
model = utils.BiLSTMModel(input_size=len(feature_cols)).to(device)
model.load_state_dict(torch.load('models/lstm_best.pth'))
model.eval()

raw = pd.read_csv('data/final_dataset.csv', parse_dates=['date'])
raw = raw.sort_values('date').reset_index(drop=True)
n = len(raw)
start = n - HORIZON
actual = raw[TARGET].values.copy()
dates = raw['date'].values


def predict_diff(window_df):
    """对给定的 SEQ_LEN 行特征窗口预测差分值（原始尺度）。"""
    X = window_df[feature_cols].values
    X_s = scaler_X.transform(X)
    x = torch.FloatTensor(X_s).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_s = model(x).cpu().numpy()
    return float(scaler_y.inverse_transform(pred_s).flatten()[0])


# ============================================================
# 1. 递归多步预测（预测值回灌）
# ============================================================
work = raw.copy()
recursive_pred = np.full(n, np.nan)
for r in range(start, n):
    feat = utils.add_diff_features(work)
    window = feat.iloc[r - SEQ_LEN:r]          # 行 r-28..r-1
    diff_hat = predict_diff(window)
    prev_val = work.at[r - 1, TARGET]          # 递归：用上一日(可能是预测值)
    y_hat = prev_val + diff_hat
    work.at[r, TARGET] = y_hat                 # 回灌预测
    recursive_pred[r] = y_hat

# ============================================================
# 2. 单步预测对比（始终用真实前一天值）
# ============================================================
feat_actual = utils.add_diff_features(raw)
onestep_pred = np.full(n, np.nan)
for r in range(start, n):
    window = feat_actual.iloc[r - SEQ_LEN:r]
    diff_hat = predict_diff(window)
    onestep_pred[r] = actual[r - 1] + diff_hat

# ============================================================
# 3. 评估与可视化
# ============================================================
truth_h = actual[start:n]
rec_h = recursive_pred[start:n]
one_h = onestep_pred[start:n]

m_rec = utils.compute_metrics(rec_h, truth_h)
m_one = utils.compute_metrics(one_h, truth_h)
print(f"递归多步({HORIZON}天): RMSE {m_rec['RMSE']:.1f} | MAPE {m_rec['MAPE']:.2f}% | R² {m_rec['R2']:.4f}")
print(f"单步对照({HORIZON}天): RMSE {m_one['RMSE']:.1f} | MAPE {m_one['MAPE']:.2f}% | R² {m_one['R2']:.4f}")

fig, ax = plt.subplots(figsize=(14, 6))
hist_start = start - 60
ax.plot(dates[hist_start:start], actual[hist_start:start],
        color='gray', linewidth=1.5, label='历史真实值')
ax.plot(dates[start:n], truth_h, color='black', linewidth=2.5, marker='o', ms=3, label='真实值')
ax.plot(dates[start:n], one_h, color='#27AE60', linewidth=2, linestyle='--',
        label=f"单步预测 (MAPE={m_one['MAPE']:.2f}%)")
ax.plot(dates[start:n], rec_h, color='#E74C3C', linewidth=2, linestyle='--',
        label=f"递归{HORIZON}天预测 (MAPE={m_rec['MAPE']:.2f}%)")
ax.axvline(dates[start], color='blue', linestyle=':', alpha=0.6, label='预测起点')
ax.set_title(f'{BEST_NAME} 未来 {HORIZON} 天电力需求预测（递归多步 vs 单步）',
             fontsize=14, fontweight='bold')
ax.set_xlabel('日期'); ax.set_ylabel('日总用电量 (MWh)')
ax.legend(fontsize=10, loc='upper left'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/22_future_forecast.png', dpi=150, bbox_inches='tight')
plt.close()

out = pd.DataFrame({
    'date': dates[start:n],
    'truth': truth_h,
    'onestep_pred': one_h,
    'recursive_pred': rec_h,
})
out.to_csv('data/future_forecast.csv', index=False)
pd.DataFrame([
    {'mode': f'单步({HORIZON}天)', **m_one},
    {'mode': f'递归多步({HORIZON}天)', **m_rec},
]).to_csv('data/future_forecast_metrics.csv', index=False)

print("✅ 图22: 未来预测已保存")
print("✅ data/future_forecast.csv / future_forecast_metrics.csv 已保存")
