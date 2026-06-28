"""
step17_schematics.py
生成报告中两张示意图：
  图2-1 数据获取与整合流程示意图  -> figures/27_data_flow.png
  图5-1 整体建模框架示意图        -> figures/28_model_framework.png
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('figures', exist_ok=True)


def box(ax, xy, w, h, text, fc, ec, fs=11, bold=False):
    x, y = xy
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', zorder=3, wrap=True)


def arrow(ax, p1, p2, color='#555555'):
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=16,
                        linewidth=1.6, color=color, zorder=1,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ============================================================
# 图2-1 数据获取与整合流程
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis('off')

BLUE, GREEN, ORANGE, GRAY, PURPLE = '#D6E4F0', '#D9EAD3', '#FCE5CD', '#EAEAEA', '#E6D6F0'
EBLUE, EGREEN, EORANGE, EGRAY, EPURPLE = '#2E5496', '#5C8A3A', '#C77A30', '#888888', '#7A4F9E'

# 三个数据源
box(ax, (2, 8.5), 3.2, 1.3, 'Kaggle AEP\n小时级电力负荷\n(2004–2018)', BLUE, EBLUE, 10)
box(ax, (6, 8.5), 3.2, 1.3, 'Open-Meteo API\nColumbus 日度气象\n(温/湿/压/风/降水/日照)', GREEN, EGREEN, 10)
box(ax, (10, 8.5), 3.2, 1.3, 'US 联邦节假日库\n+ 派生时间特征', ORANGE, EORANGE, 10)

# 处理层
box(ax, (2, 6), 3.2, 1.1, '小时聚合为日度\n(step3)', GRAY, EGRAY, 10)
box(ax, (6, 6), 3.2, 1.1, '清洗/缺失值处理\n时间对齐', GRAY, EGRAY, 10)
box(ax, (10, 6), 3.2, 1.1, '节假日/星期/季节\n编码 (step2)', GRAY, EGRAY, 10)

# 合并
box(ax, (6, 3.6), 5.2, 1.2, '多源数据按日期合并 (step4)\n2004-10-02 ~ 2018-08-02，共 5053 天', PURPLE, EPURPLE, 11, bold=True)

# 最终数据集
box(ax, (6, 1.2), 6.4, 1.2, '最终数据集 final_dataset.csv\n5 大维度 / 28 维原始特征', BLUE, EBLUE, 11, bold=True)

# 箭头
arrow(ax, (2, 7.85), (2, 6.55))
arrow(ax, (6, 7.85), (6, 6.55))
arrow(ax, (10, 7.85), (10, 6.55))
arrow(ax, (2, 5.45), (4.6, 4.2))
arrow(ax, (6, 5.45), (6, 4.2))
arrow(ax, (10, 5.45), (7.4, 4.2))
arrow(ax, (6, 3.0), (6, 1.8))

ax.set_title('数据获取与整合流程', fontsize=14, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('figures/27_data_flow.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ 图2-1 已生成 figures/27_data_flow.png')


# ============================================================
# 图5-1 整体建模框架
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 11); ax.axis('off')

# 输入
box(ax, (6, 10), 7.5, 1.0, '最终数据集（电力 + 气象 + 时间 + 节假日）', BLUE, EBLUE, 11, bold=True)

# 特征工程
box(ax, (6, 8.3), 9.2, 1.1,
    '特征工程：滞后 lag · 滑动统计 rolling · 温度衍生 CDD/HDD · 周期编码 sin/cos · 季节 One-Hot',
    GREEN, EGREEN, 9.5)

# 差分 + 序列化 + 标准化
box(ax, (2.3, 6.4), 3.0, 1.0, '差分目标\nΔy = y(t) − y(t-1)', ORANGE, EORANGE, 10)
box(ax, (6, 6.4), 3.0, 1.0, '滑动窗口序列化\nSEQ_LEN = 28', ORANGE, EORANGE, 10)
box(ax, (9.7, 6.4), 3.0, 1.0, '标准化\nStandardScaler', ORANGE, EORANGE, 10)

# 三个模型
box(ax, (2.3, 4.3), 3.0, 1.1, 'BiLSTM\n(step7)', PURPLE, EPURPLE, 11, bold=True)
box(ax, (6, 4.3), 3.0, 1.1, 'BiGRU\n(step8)', PURPLE, EPURPLE, 11, bold=True)
box(ax, (9.7, 4.3), 3.0, 1.1, 'CNN-LSTM\n(step9)', PURPLE, EPURPLE, 11, bold=True)

# 还原
box(ax, (6, 2.5), 5.6, 1.0, '差分还原：y_hat(t) = y(t-1) + Δy_hat', GRAY, EGRAY, 11, bold=True)

# 评估
box(ax, (6, 0.8), 8.6, 1.0,
    '评估与对比：RMSE / MAE / MAPE / R²  +  基线对比 + 消融 + 交叉验证', BLUE, EBLUE, 10, bold=True)

# 箭头
arrow(ax, (6, 9.5), (6, 8.85))
arrow(ax, (6, 7.75), (6, 6.9))
arrow(ax, (4.5, 7.75), (2.3, 6.9))
arrow(ax, (7.5, 7.75), (9.7, 6.9))
for x in (2.3, 6, 9.7):
    arrow(ax, (x, 5.9), (x, 4.85))
for x in (2.3, 9.7):
    arrow(ax, (x, 3.75), (6, 3.0))
arrow(ax, (6, 3.75), (6, 3.0))
arrow(ax, (6, 2.0), (6, 1.3))

ax.set_title('整体建模框架', fontsize=14, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('figures/28_model_framework.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ 图5-1 已生成 figures/28_model_framework.png')
