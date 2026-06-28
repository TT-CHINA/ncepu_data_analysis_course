"""
step14_cross_validation.py
滚动起点交叉验证（Rolling-Origin Cross-Validation）。

目的：检验"线性回归优于深度模型"这一结论是否跨时间稳健，而非单一测试集的偶然。
做法：采用扩展窗口（expanding window），划分 4 个连续的时间折，每折：
      训练 = 折起点之前的全部数据，验证 = 训练区末段 1 年，测试 = 之后 1 年。
      在每折上以完全相同的特征与流程评估 LR / BiLSTM / BiGRU / CNN-LSTM，
      汇总各模型 MAPE、R² 的 mean ± std。
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('figures', exist_ok=True)

device = torch.device('cpu')
SEQ_LEN = utils.SEQ_LEN
TARGET = utils.TARGET
TEST_SIZE = 365
VAL_SIZE = 365
N_FOLDS = 4
EPOCHS = 200
PATIENCE = 30

df_clean, feature_cols = utils.build_diff_frame()
N = len(df_clean)
prev_all = df_clean[TARGET].shift(1).fillna(df_clean[TARGET].iloc[0]).values
truth_all = df_clean[TARGET].values
X_all = df_clean[feature_cols].values
ydiff_all = df_clean[['target_diff']].values

MODELS = {
    'BiLSTM': utils.BiLSTMModel,
    'BiGRU': utils.BiGRUModel,
    'CNN-LSTM': utils.CNNLSTMModel,
}


def train_dl(ModelCls, tr, va, te, scaler_y):
    """训练一个深度模型并返回测试集还原后的预测。tr/va/te 为 (X_s, y_s, prev, truth)。"""
    utils.set_seed()
    Xtr, ytr, _, _ = utils.create_sequences_with_prev(*tr, SEQ_LEN)
    Xva, yva, _, _ = utils.create_sequences_with_prev(*va, SEQ_LEN)
    Xte, yte, prev_te, truth_te = utils.create_sequences_with_prev(*te, SEQ_LEN)
    trl = utils.make_loader(Xtr, ytr, 32, True)
    val = utils.make_loader(Xva, yva, 32, False)
    tel = utils.make_loader(Xte, yte, 32, False)

    model = ModelCls(input_size=len(feature_cols)).to(device)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-5)
    best, cnt, best_state = float('inf'), 0, None
    for ep in range(EPOCHS):
        model.train()
        for Xb, yb in trl:
            opt.zero_grad()
            loss = crit(model(Xb.to(device)), yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vl = 0
        with torch.no_grad():
            for Xb, yb in val:
                vl += crit(model(Xb.to(device)), yb.to(device)).item() * Xb.size(0)
        vl /= len(val.dataset)
        sch.step()
        if vl < best:
            best, cnt = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            cnt += 1
            if cnt >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in tel:
            preds.append(model(Xb.to(device)).cpu().numpy())
    pred_diff = scaler_y.inverse_transform(np.vstack(preds)).flatten()
    return prev_te + pred_diff, truth_te


records = []
for fold in range(N_FOLDS):
    test_end = N - (N_FOLDS - 1 - fold) * TEST_SIZE
    test_start = test_end - TEST_SIZE
    val_start = test_start - VAL_SIZE
    sl_tr = slice(0, val_start)
    sl_va = slice(val_start, test_start)
    sl_te = slice(test_start, test_end)

    scaler_X = StandardScaler().fit(X_all[sl_tr])
    scaler_y = StandardScaler().fit(ydiff_all[sl_tr])

    def pack(sl):
        return (scaler_X.transform(X_all[sl]), scaler_y.transform(ydiff_all[sl]),
                prev_all[sl], truth_all[sl])
    tr, va, te = pack(sl_tr), pack(sl_va), pack(sl_te)

    d0 = df_clean['date'].iloc[test_start].date()
    d1 = df_clean['date'].iloc[test_end - 1].date()
    print(f"\n===== 折{fold+1}: 训练{val_start}天 | 测试 {d0}~{d1} =====")

    # 线性回归
    lr = LinearRegression().fit(tr[0], tr[1].ravel())
    lr_diff = scaler_y.inverse_transform(
        lr.predict(te[0][SEQ_LEN:]).reshape(-1, 1)).flatten()
    lr_pred = te[2][SEQ_LEN:] + lr_diff
    lr_truth = te[3][SEQ_LEN:]
    m = utils.compute_metrics(lr_pred, lr_truth)
    records.append({'fold': fold + 1, 'model': 'Linear Regression', **m})
    print(f"  Linear Regression : MAPE {m['MAPE']:.2f}% | R² {m['R2']:.3f}")

    # 深度模型
    for name, cls in MODELS.items():
        t0 = time.time()
        pred, truth = train_dl(cls, tr, va, te, scaler_y)
        m = utils.compute_metrics(pred, truth)
        records.append({'fold': fold + 1, 'model': name, **m})
        print(f"  {name:10s}: MAPE {m['MAPE']:.2f}% | R² {m['R2']:.3f} | {time.time()-t0:.0f}s")

cv = pd.DataFrame(records)
cv.to_csv('data/cv_results.csv', index=False)

order = ['Linear Regression', 'BiLSTM', 'BiGRU', 'CNN-LSTM']
summary = cv.groupby('model').agg(
    MAPE_mean=('MAPE', 'mean'), MAPE_std=('MAPE', 'std'),
    R2_mean=('R2', 'mean'), R2_std=('R2', 'std')).reindex(order).reset_index()
summary.to_csv('data/cv_summary.csv', index=False)

print("\n" + "=" * 60)
print("📊 滚动起点交叉验证汇总（4 折，mean ± std）")
print("=" * 60)
for _, r in summary.iterrows():
    print(f"{r['model']:18s} | MAPE {r['MAPE_mean']:.2f}% ± {r['MAPE_std']:.2f} | "
          f"R² {r['R2_mean']:.3f} ± {r['R2_std']:.3f}")

# 图：各模型 MAPE mean ± std
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#f39c12', '#3498db', '#9b59b6', '#27AE60']
bars = ax.bar(summary['model'], summary['MAPE_mean'],
              yerr=summary['MAPE_std'], capsize=6,
              color=colors, edgecolor='black', alpha=0.9)
for b, mn, sd in zip(bars, summary['MAPE_mean'], summary['MAPE_std']):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + sd + 0.05,
            f'{mn:.2f}±{sd:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('MAPE (%) — 越低越好', fontsize=12)
ax.set_title('滚动起点交叉验证（4 折）各模型 MAPE 对比', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figures/23_cv_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ 图23 / data/cv_results.csv / data/cv_summary.csv 已保存")
