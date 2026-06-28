"""
step13_diff_vs_original.py
消融实验：验证"差分预测策略"的有效性。
在相同的 BiLSTM 结构上对比三种方案：
  (1) 原始目标       —— 直接预测 y(t)
  (2) 原始目标+Bias校正 —— 用验证集均值偏差校正
  (3) 差分目标       —— 预测 Δy(t)，复用 step7 结果
结果写入 data/diff_vs_original.csv，对应报告表 6-2。
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

utils.set_seed()
device = torch.device('cpu')
SEQ_LEN = utils.SEQ_LEN
TARGET = utils.TARGET

# ============================================================
# 原始目标数据准备（不做差分）
# ============================================================
df_clean, feature_cols = utils.build_diff_frame()
train, val, test = utils.time_split(df_clean)

scaler_X = StandardScaler().fit(train[feature_cols].values)
scaler_y = StandardScaler().fit(train[[TARGET]].values)


def make_seq(part):
    X = scaler_X.transform(part[feature_cols].values)
    y = scaler_y.transform(part[[TARGET]].values)
    truth = part[TARGET].values
    Xs, ys, _, ts = utils.create_sequences_with_prev(X, y, truth, truth, SEQ_LEN)
    return Xs, ys, ts


X_tr, y_tr, _ = make_seq(train)
X_va, y_va, truth_va = make_seq(val)
X_te, y_te, truth_te = make_seq(test)

train_loader = utils.make_loader(X_tr, y_tr, 32, shuffle=True)
val_loader = utils.make_loader(X_va, y_va, 32, shuffle=False)
test_loader = utils.make_loader(X_te, y_te, 32, shuffle=False)

# ============================================================
# 训练原始目标 BiLSTM
# ============================================================
model = utils.BiLSTMModel(input_size=len(feature_cols)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5)

best_val, patience, counter = float('inf'), 40, 0
print("训练原始目标 BiLSTM...")
for epoch in range(200):
    model.train()
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb.to(device)), yb.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    vl = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            vl += criterion(model(Xb.to(device)), yb.to(device)).item() * Xb.size(0)
    vl /= len(val_loader.dataset)
    scheduler.step()
    if vl < best_val:
        best_val, counter = vl, 0
        torch.save(model.state_dict(), 'models/lstm_original.pth')
    else:
        counter += 1
    if counter >= patience:
        print(f"早停于 Epoch {epoch+1}")
        break

model.load_state_dict(torch.load('models/lstm_original.pth'))
model.eval()


def predict(loader):
    out = []
    with torch.no_grad():
        for Xb, _ in loader:
            out.append(model(Xb.to(device)).cpu().numpy())
    return scaler_y.inverse_transform(np.vstack(out)).flatten()


pred_val = predict(val_loader)
pred_test = predict(test_loader)

# 方案1：原始目标
m_orig = utils.compute_metrics(pred_test, truth_te)
# 方案2：原始目标 + Bias校正（用验证集均值偏差）
bias = float(np.mean(truth_va - pred_val))
m_bias = utils.compute_metrics(pred_test + bias, truth_te)
# 方案3：差分目标（复用 step7 结果）
lstm_diff = pd.read_csv('data/lstm_metrics.csv').iloc[0]
m_diff = {'RMSE': float(lstm_diff['RMSE']), 'MAE': float(lstm_diff['MAE']),
          'MAPE': float(lstm_diff['MAPE']), 'R2': float(lstm_diff['R2'])}

table = pd.DataFrame([
    {'策略': '原始目标', 'MAPE': m_orig['MAPE'], 'R2': m_orig['R2'],
     '预测均值偏差': float(pred_test.mean() - truth_te.mean())},
    {'策略': '原始目标 + Bias校正', 'MAPE': m_bias['MAPE'], 'R2': m_bias['R2'],
     '预测均值偏差': float((pred_test + bias).mean() - truth_te.mean())},
    {'策略': '差分目标', 'MAPE': m_diff['MAPE'], 'R2': m_diff['R2'],
     '预测均值偏差': float(pd.read_csv('data/lstm_results.csv').eval('lstm_pred - truth').mean())},
])
table.to_csv('data/diff_vs_original.csv', index=False)
print("\n差分 vs 原始目标 对比（对应报告表6-2）:")
print(table.to_string(index=False))
print("\n✅ data/diff_vs_original.csv 已保存")
