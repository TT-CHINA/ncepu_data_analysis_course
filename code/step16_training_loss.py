"""
step16_training_loss.py
统一训练 BiLSTM / BiGRU / CNN-LSTM 三个模型并记录训练-验证损失曲线，
生成图6-1「三模型训练损失曲线对比」。超参数与 step7~9 保持一致。
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

utils.set_seed()
device = torch.device('cpu')
os.makedirs('figures', exist_ok=True)

# 数据准备（与 step7 完全一致）
data = utils.prepare_diff_data()
SEQ = utils.SEQ_LEN
Xtr, ytr, _, _ = utils.create_sequences_with_prev(
    data['X_train_s'], data['y_train_s'], data['prev_train'], data['truth_train'], SEQ)
Xva, yva, _, _ = utils.create_sequences_with_prev(
    data['X_val_s'], data['y_val_s'], data['prev_val'], data['truth_val'], SEQ)
train_loader = utils.make_loader(Xtr, ytr, 32, shuffle=True)
val_loader = utils.make_loader(Xva, yva, 32, shuffle=False)
n_feat = Xtr.shape[2]

EPOCHS, PATIENCE = 200, 40


def train_one(model):
    utils.set_seed()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5)
    tr_hist, va_hist = [], []
    best, patience = float('inf'), 0
    for epoch in range(EPOCHS):
        model.train()
        tl = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * Xb.size(0)
        tl /= len(train_loader.dataset)
        model.eval()
        vl = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                vl += criterion(model(Xb), yb).item() * Xb.size(0)
        vl /= len(val_loader.dataset)
        tr_hist.append(tl); va_hist.append(vl)
        scheduler.step()
        if vl < best:
            best, patience = vl, 0
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    return tr_hist, va_hist


models = {
    'BiLSTM': utils.BiLSTMModel(input_size=n_feat),
    'BiGRU': utils.BiGRUModel(input_size=n_feat),
    'CNN-LSTM': utils.CNNLSTMModel(input_size=n_feat),
}
colors = {'BiLSTM': '#1f77b4', 'BiGRU': '#2ca02c', 'CNN-LSTM': '#d62728'}

histories = {}
for name, m in models.items():
    print(f'训练 {name} ...', end='', flush=True)
    t0 = time.time()
    histories[name] = train_one(m)
    print(f' 完成 {time.time()-t0:.0f}s, epochs={len(histories[name][0])}')

# 绘图：左=训练损失，右=验证损失
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for name in models:
    tr, va = histories[name]
    axes[0].plot(tr, color=colors[name], linewidth=1.8, label=name)
    axes[1].plot(va, color=colors[name], linewidth=1.8, label=name)
axes[0].set_title('三模型训练损失曲线', fontsize=13, fontweight='bold')
axes[1].set_title('三模型验证损失曲线', fontsize=13, fontweight='bold')
for ax in axes:
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE 损失（归一化）')
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/26_training_loss_compare.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ 图6-1 已生成 figures/26_training_loss_compare.png')
