"""
step7_lstm_diff.py
LSTM 差分预测版 - 预测 Δy 而非 y
（数据准备、模型结构、序列化等公共逻辑统一来自 utils.py）
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

utils.set_seed()
device = torch.device('cpu')

os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ============================================================
# 1. 数据准备（差分目标）
# ============================================================
data = utils.prepare_diff_data()
feature_cols = data['feature_cols']
print(f"特征数: {len(feature_cols)}")

train, val, test = data['train'], data['val'], data['test']
print(f"训练集: {len(train)} | 验证集: {len(val)} | 测试集: {len(test)}")

SEQ_LEN = utils.SEQ_LEN
X_train_seq, y_train_seq, prev_train_seq, truth_train_seq = utils.create_sequences_with_prev(
    data['X_train_s'], data['y_train_s'], data['prev_train'], data['truth_train'], SEQ_LEN)
X_val_seq, y_val_seq, prev_val_seq, truth_val_seq = utils.create_sequences_with_prev(
    data['X_val_s'], data['y_val_s'], data['prev_val'], data['truth_val'], SEQ_LEN)
X_test_seq, y_test_seq, prev_test_seq, truth_test_seq = utils.create_sequences_with_prev(
    data['X_test_s'], data['y_test_s'], data['prev_test'], data['truth_test'], SEQ_LEN)

print(f"X_train_seq: {X_train_seq.shape}")

train_loader = utils.make_loader(X_train_seq, y_train_seq, 32, shuffle=True)
val_loader = utils.make_loader(X_val_seq, y_val_seq, 32, shuffle=False)
test_loader = utils.make_loader(X_test_seq, y_test_seq, 32, shuffle=False)

# ============================================================
# 2. 模型与训练
# ============================================================
model = utils.BiLSTMModel(input_size=X_train_seq.shape[2]).to(device)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-5)

EPOCHS = 200
PATIENCE = 40
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience_counter = 0

print("\n开始训练...")
start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_loss += criterion(model(X_batch), y_batch).item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/lstm_best.pth')
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    if patience_counter >= PATIENCE:
        print(f"\n⏹ 早停于 Epoch {epoch+1}")
        break

training_time = time.time() - start_time
print(f"训练用时 {training_time:.1f} 秒")

# ============================================================
# 3. 评估
# ============================================================
model.load_state_dict(torch.load('models/lstm_best.pth'))
model.eval()

test_preds_diff = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        test_preds_diff.append(model(X_batch.to(device)).cpu().numpy())
test_preds_diff = np.vstack(test_preds_diff)

preds_diff = data['scaler_y'].inverse_transform(test_preds_diff).flatten()
preds_absolute = prev_test_seq + preds_diff
truths = truth_test_seq

m = utils.compute_metrics(preds_absolute, truths)
print("\n" + "=" * 60)
print("📊 LSTM 差分预测 测试集评估")
print("=" * 60)
print(f"RMSE:  {m['RMSE']:,.2f} MWh")
print(f"MAE:   {m['MAE']:,.2f} MWh")
print(f"MAPE:  {m['MAPE']:.2f}%")
print(f"R²:    {m['R2']:.4f}")

# ============================================================
# 4. 可视化与保存
# ============================================================
test_dates = test['date'].reset_index(drop=True)
test_dates_seq = test_dates.iloc[SEQ_LEN:].reset_index(drop=True)

fig, axes = plt.subplots(2, 1, figsize=(14, 9))
axes[0].plot(train_losses, label='训练损失')
axes[0].plot(val_losses, label='验证损失')
axes[0].set_title('LSTM 差分预测 训练损失')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(test_dates_seq, truths, label='真实值', linewidth=2, color='steelblue')
axes[1].plot(test_dates_seq, preds_absolute, label='预测值', linewidth=2, linestyle='--', color='red')
axes[1].set_title(f"LSTM 差分预测 (RMSE={m['RMSE']:.0f}, MAPE={m['MAPE']:.2f}%, R²={m['R2']:.3f})")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/11_lstm_diff_prediction.png', dpi=150, bbox_inches='tight')
plt.close()

pd.DataFrame({'date': test_dates_seq, 'truth': truths, 'lstm_pred': preds_absolute}).to_csv(
    'data/lstm_results.csv', index=False)
pd.DataFrame([{'model': 'LSTM (Differencing)', **m,
               'training_time': training_time,
               'params': sum(p.numel() for p in model.parameters())}]).to_csv(
    'data/lstm_metrics.csv', index=False)

# 保存差分中间数据，供 GRU / CNN-LSTM 复用
np.savez('data/diff_data.npz',
         X_train=data['X_train_s'], y_train_diff=data['y_train_s'],
         X_val=data['X_val_s'], y_val_diff=data['y_val_s'],
         X_test=data['X_test_s'], y_test_diff=data['y_test_s'],
         prev_train=data['prev_train'], prev_val=data['prev_val'], prev_test=data['prev_test'],
         truth_train=data['truth_train'], truth_val=data['truth_val'], truth_test=data['truth_test'])
joblib.dump(data['scaler_y'], 'models/scaler_y_diff.pkl')
joblib.dump(data['scaler_X'], 'models/scaler_X_diff.pkl')
# 测试集日期供后续脚本使用
test[['date']].to_csv('data/test_dates.csv', index=False)

print("\n✅ 完成")
