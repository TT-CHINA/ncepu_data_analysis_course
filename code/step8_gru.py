"""
step8_gru.py
GRU 差分预测 - 与 LSTM 对照实验（公共逻辑来自 utils.py）
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

# 加载与 LSTM 差分版相同的数据
data = np.load('data/diff_data.npz')
scaler_y = joblib.load('models/scaler_y_diff.pkl')
test_dates = pd.read_csv('data/test_dates.csv', parse_dates=['date'])

SEQ_LEN = utils.SEQ_LEN
X_train_seq, y_train_seq, _, _ = utils.create_sequences_with_prev(
    data['X_train'], data['y_train_diff'], data['prev_train'], data['truth_train'], SEQ_LEN)
X_val_seq, y_val_seq, _, _ = utils.create_sequences_with_prev(
    data['X_val'], data['y_val_diff'], data['prev_val'], data['truth_val'], SEQ_LEN)
X_test_seq, y_test_seq, prev_test_seq, truth_test_seq = utils.create_sequences_with_prev(
    data['X_test'], data['y_test_diff'], data['prev_test'], data['truth_test'], SEQ_LEN)

print(f"X_train_seq: {X_train_seq.shape}")

train_loader = utils.make_loader(X_train_seq, y_train_seq, 32, shuffle=True)
val_loader = utils.make_loader(X_val_seq, y_val_seq, 32, shuffle=False)
test_loader = utils.make_loader(X_test_seq, y_test_seq, 32, shuffle=False)

model = utils.BiGRUModel(input_size=X_train_seq.shape[2]).to(device)
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

print("\n开始训练 GRU...")
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
        torch.save(model.state_dict(), 'models/gru_best.pth')
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    if patience_counter >= PATIENCE:
        print(f"\n⏹ 早停于 Epoch {epoch+1}")
        break

training_time = time.time() - start_time
print(f"训练用时 {training_time:.1f} 秒")

model.load_state_dict(torch.load('models/gru_best.pth'))
model.eval()
test_preds_diff = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        test_preds_diff.append(model(X_batch.to(device)).cpu().numpy())
test_preds_diff = np.vstack(test_preds_diff)

preds_diff = scaler_y.inverse_transform(test_preds_diff).flatten()
preds_absolute = prev_test_seq + preds_diff
truths = truth_test_seq

m = utils.compute_metrics(preds_absolute, truths)
print("\n" + "=" * 60)
print("📊 GRU 差分预测 测试集评估")
print("=" * 60)
print(f"RMSE:  {m['RMSE']:,.2f} MWh | MAE: {m['MAE']:,.2f} | MAPE: {m['MAPE']:.2f}% | R²: {m['R2']:.4f}")
print(f"训练用时: {training_time:.1f} 秒")

test_dates_seq = test_dates['date'].iloc[SEQ_LEN:].reset_index(drop=True)
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
axes[0].plot(train_losses, label='训练损失')
axes[0].plot(val_losses, label='验证损失')
axes[0].set_title('GRU 训练损失曲线', fontsize=13, fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(test_dates_seq, truths, label='真实值', linewidth=2, color='steelblue')
axes[1].plot(test_dates_seq, preds_absolute, label='GRU预测', linewidth=2, linestyle='--', color='green')
axes[1].set_title(f"GRU 测试集预测 (RMSE={m['RMSE']:.0f}, MAPE={m['MAPE']:.2f}%, R²={m['R2']:.3f})",
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('日期'); axes[1].set_ylabel('日总用电量 (MWh)')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/13_gru_prediction.png', dpi=150, bbox_inches='tight')
plt.close()

pd.DataFrame({'date': test_dates_seq, 'truth': truths, 'gru_pred': preds_absolute}).to_csv(
    'data/gru_results.csv', index=False)
pd.DataFrame([{'model': 'GRU (Differencing)', **m,
               'training_time': training_time,
               'params': sum(p.numel() for p in model.parameters())}]).to_csv(
    'data/gru_metrics.csv', index=False)

print("\n✅ GRU 完成")
