"""
step15_temp_ushape.py
温度-电力 U 型关系的量化建模

在 EDA（图5）发现 U 型关系的基础上，进一步量化该非线性关系：
  1) 对比三种温度-电力模型：线性 / 二次多项式 / 分段度日（HDD-CDD）
  2) 搜索最优平衡点温度 T_b（用电量谷值对应的"舒适温度"）
  3) 量化制冷侧、采暖侧的边际用电敏感度（MWh / °C）
输出：figures/24_temp_ushape_model.png、data/temp_ushape_metrics.csv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# 配置中文字体（与 step5 保持一致）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 读取数据
df = pd.read_csv('data/final_dataset.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
T = df['tavg'].values
y = df['electricity_total'].values
print(f"数据加载完成：{df.shape}，温度范围 {T.min():.1f}~{T.max():.1f}°C")


# ============================================================
# 模型1：线性  electricity ~ a + b*T
# ============================================================
lin = LinearRegression().fit(T.reshape(-1, 1), y)
y_lin = lin.predict(T.reshape(-1, 1))
r2_lin = r2_score(y, y_lin)
print(f"\n[线性]   R²={r2_lin:.4f}  斜率={lin.coef_[0]:.1f} MWh/°C")


# ============================================================
# 模型2：二次多项式  electricity ~ a + b*T + c*T²
# ============================================================
z = np.polyfit(T, y, 2)
p2 = np.poly1d(z)
y_quad = p2(T)
r2_quad = r2_score(y, y_quad)
# 抛物线顶点（谷值温度）：T* = -b / (2c)
t_vertex = -z[1] / (2 * z[0])
print(f"[二次]   R²={r2_quad:.4f}  谷值温度={t_vertex:.1f}°C")


# ============================================================
# 模型3：分段度日模型（HDD-CDD）
#   给定平衡点 T_b:  HDD = max(T_b - T, 0), CDD = max(T - T_b, 0)
#   electricity ~ a + b_h*HDD + b_c*CDD
#   网格搜索 T_b 使 R² 最大 —— b_h 为采暖敏感度, b_c 为制冷敏感度
# ============================================================
best = {'r2': -np.inf}
for tb in np.arange(5.0, 25.0, 0.5):
    hdd = np.maximum(tb - T, 0)
    cdd = np.maximum(T - tb, 0)
    X = np.column_stack([hdd, cdd])
    m = LinearRegression().fit(X, y)
    r2 = r2_score(y, m.predict(X))
    if r2 > best['r2']:
        best = {'r2': r2, 'tb': tb, 'b_h': m.coef_[0], 'b_c': m.coef_[1],
                'a': m.intercept_, 'model': m}

tb = best['tb']
hdd = np.maximum(tb - T, 0)
cdd = np.maximum(T - tb, 0)
y_seg = best['model'].predict(np.column_stack([hdd, cdd]))
r2_seg = best['r2']
print(f"[分段]   R²={r2_seg:.4f}  平衡点 T_b={tb:.1f}°C")
print(f"         采暖敏感度 b_h={best['b_h']:.0f} MWh/°C（温度每降1°C的增量）")
print(f"         制冷敏感度 b_c={best['b_c']:.0f} MWh/°C（温度每升1°C的增量）")
print(f"         基荷 a={best['a']:.0f} MWh")


# ============================================================
# 保存指标
# ============================================================
metrics = pd.DataFrame([
    {'模型': '线性', 'R2': r2_lin, '谷值/平衡点(°C)': np.nan,
     '采暖敏感度(MWh/°C)': np.nan, '制冷敏感度(MWh/°C)': np.nan, '参数个数': 2},
    {'模型': '二次多项式', 'R2': r2_quad, '谷值/平衡点(°C)': t_vertex,
     '采暖敏感度(MWh/°C)': np.nan, '制冷敏感度(MWh/°C)': np.nan, '参数个数': 3},
    {'模型': '分段度日(HDD-CDD)', 'R2': r2_seg, '谷值/平衡点(°C)': tb,
     '采暖敏感度(MWh/°C)': best['b_h'], '制冷敏感度(MWh/°C)': best['b_c'], '参数个数': 3},
])
metrics.to_csv('data/temp_ushape_metrics.csv', index=False, encoding='utf-8-sig')
print("\n✅ 指标已保存 data/temp_ushape_metrics.csv")


# ============================================================
# 可视化（与现有图表统一风格）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左图：三种拟合曲线叠加在散点上
order = np.argsort(T)
Ts = T[order]
axes[0].scatter(T, y, s=8, alpha=0.25, color='steelblue', label='观测样本')
axes[0].plot(Ts, y_lin[order], color='green', linewidth=2, linestyle=':', label=f'线性 (R²={r2_lin:.3f})')
axes[0].plot(Ts, y_quad[order], color='black', linewidth=2, linestyle='--', label=f'二次 (R²={r2_quad:.3f})')
axes[0].plot(Ts, y_seg[order], color='red', linewidth=2.5, label=f'分段度日 (R²={r2_seg:.3f})')
axes[0].axvline(tb, color='red', linewidth=1.2, alpha=0.6)
axes[0].text(tb + 0.4, axes[0].get_ylim()[1] * 0.97,
             f'平衡点 {tb:.1f}°C', color='red', fontsize=11, va='top', fontweight='bold')
axes[0].set_title('温度-电力 U 型关系的三种量化模型', fontsize=13, fontweight='bold')
axes[0].set_xlabel('日均温度 (°C)')
axes[0].set_ylabel('日总用电量 (MWh)')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# 右图：分段度日模型的采暖/制冷敏感度示意
axes[1].scatter(T, y, s=8, alpha=0.2, color='gray')
# 采暖段（T<T_b）红蓝双色拟合直线
left = Ts[Ts <= tb]
right = Ts[Ts >= tb]
y_base = best['a']
axes[1].plot(left, y_base + best['b_h'] * (tb - left), color='#1f77b4', linewidth=3,
             label=f'采暖段 斜率 {best["b_h"]:.0f} MWh/°C')
axes[1].plot(right, y_base + best['b_c'] * (right - tb), color='#d62728', linewidth=3,
             label=f'制冷段 斜率 {best["b_c"]:.0f} MWh/°C')
axes[1].axvline(tb, color='black', linestyle='--', linewidth=1, alpha=0.6)
axes[1].scatter([tb], [y_base], color='black', zorder=5, s=60)
axes[1].text(tb + 0.4, y_base, f'谷底基荷\n{y_base:.0f} MWh', fontsize=10, va='center')
axes[1].set_title('采暖 vs 制冷的边际用电敏感度', fontsize=13, fontweight='bold')
axes[1].set_xlabel('日均温度 (°C)')
axes[1].set_ylabel('日总用电量 (MWh)')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/24_temp_ushape_model.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图24: 温度 U 型量化模型")

# ============================================================
# 关键结论输出
# ============================================================
print("\n" + "=" * 60)
print("📊 温度 U 型量化结论（可写入报告）")
print("=" * 60)
print(f"1. 二次顶点谷值温度: {t_vertex:.1f}°C")
print(f"2. 分段度日最优平衡点 T_b: {tb:.1f}°C（R²={r2_seg:.3f}）")
print(f"3. 采暖敏感度: 温度每降低 1°C，日用电增加约 {best['b_h']:.0f} MWh")
print(f"4. 制冷敏感度: 温度每升高 1°C，日用电增加约 {best['b_c']:.0f} MWh")
ratio = best['b_c'] / best['b_h']
hotter = '制冷' if ratio > 1 else '采暖'
print(f"5. 制冷/采暖敏感度比 = {ratio:.2f} → {hotter}侧更陡（夏季空调负荷更敏感）" if ratio > 1
      else f"5. 制冷/采暖敏感度比 = {ratio:.2f} → 采暖侧更陡")
print(f"6. R² 提升: 线性 {r2_lin:.3f} → 二次 {r2_quad:.3f} → 分段 {r2_seg:.3f}")
print(f"   说明温度对电力的影响是显著非线性的，单一线性项不足以刻画。")
