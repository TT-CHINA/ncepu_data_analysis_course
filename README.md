# 电力消费趋势分析与预测

> 数据科学课程设计 · 题目1：**电力消费趋势分析与预测**
> 基于多源数据（电力 + 气象 + 节假日 + 时间特征）的日度电力消费深度学习预测系统。

---

## 一、研究背景与业务理解

电力消费具有显著的时序波动特征，受 **温度、节假日、星期效应、季节** 等多重因素影响。
准确预测日度电力消费可用于：

- **电网调度**：提前安排发电与购电计划，降低弃风弃光。
- **负荷管理**：识别峰谷规律，引导错峰用电。
- **能源经济**：辅助电价制定与碳排放管理。

本项目以 **美国 AEP 服务区域日度电力总消费量（MWh）** 为预测目标，选取 AEP 服务区域内的 **Columbus, Ohio** 作为代表气象站点，构建并对比 **3 种深度学习模型**（LSTM / GRU / CNN-LSTM），定量分析各类因素对电力消费的影响。

---

## 二、数据获取

| 数据源 | 内容 | 时间范围 | 维度 |
| --- | --- | --- | --- |
| **Kaggle: AEP Hourly Energy Consumption** ([链接](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)) | 美国电力公司 AEP 小时级负荷（MW） | 2004–2018 | 1 列 |
| **Open-Meteo Historical API** ([archive-api.open-meteo.com](https://open-meteo.com/en/docs/historical-weather-api)) | Columbus, Ohio 日度气象（温度、降水、风速、气压、湿度、日照） | 2004–2018 | 8 列 |
| **US Federal Holiday Calendar** | 美国联邦节假日（含观察日） | 2004–2018 | 2 列 |
| **派生时间特征** | 年/月/日/周/季节/周末/周期编码 | — | 10+ 列 |

最终建模数据集 **5 大维度共 46 个建模特征**：
气象因素、时间标识、周期编码、节假日、历史用电（滞后+滑动统计）。

> 数据时间窗口对齐到 **2004-10-02 ~ 2018-08-02**，共 **5053 天**（按 AEP 电力与 Columbus 气象的可用交集）。清洗（滞后/差分引入的起始 NaN）后实际建模样本 **5023 条**，按 7:1.5:1.5 划分为训练/验证/测试集（3516 / 753 / 754 天）。

---

## 三、项目结构

```
data_course/
├── data/                          # 原始与中间数据
│   ├── AEP_hourly.csv             # Kaggle 电力原始数据（需手动下载）
│   ├── columbus_weather.csv       # Columbus, Ohio 日度气象
│   ├── time_features.csv          # 时间+节假日特征
│   ├── electricity_daily.csv      # 聚合后的日度电力
│   ├── final_dataset.csv          # 合并后的最终数据集
│   ├── diff_data.npz              # 差分预测专用数据（step7 生成，下游复用）
│   ├── feature_names.txt          # 特征列表
│   ├── lstm_results.csv / gru_results.csv / cnn_lstm_results.csv
│   ├── lstm_metrics.csv / gru_metrics.csv / cnn_lstm_metrics.csv
│   ├── baseline_metrics.csv       # 基线模型指标
│   ├── future_forecast*.csv       # 递归多步预测结果与指标
│   ├── diff_vs_original.csv       # 差分 vs 原始目标消融
│   ├── cv_results.csv / cv_summary.csv # 滚动起点交叉验证结果
│   ├── feature_importance.csv     # 特征重要性
│   └── final_comparison.csv       # 三模型综合对比
├── figures/                       # 全部分析图表
├── models/                        # 训练好的模型与 scaler
├── code/
│   ├── utils.py                   # 公共模块（模型结构/特征/序列化/指标）
│   ├── step1_get_weather.py       # 拉取 Columbus 气象数据
│   ├── step2_get_holiday.py       # 生成美国联邦节假日 + 时间特征
│   ├── step3_process_electricity.py # 聚合小时数据为日度
│   ├── step4_merge_data.py        # 多源数据合并
│   ├── step5_eda.py               # 数据探索与可视化
│   ├── step6_feature_engineering.py # 特征工程 + 数据集划分演示
│   ├── step7_lstm_diff.py         # 模型1：LSTM（差分预测版）
│   ├── step8_gru.py               # 模型2：GRU
│   ├── step9_cnn_lstm.py          # 模型3：CNN-LSTM 混合
│   ├── step10_compare_analyze.py  # 三模型综合对比 + 影响因素分析
│   ├── step11_baseline.py         # 基线对比（持续法/周朴素/线性回归）
│   ├── step12_future_forecast.py  # 递归多步未来预测
│   ├── step13_diff_vs_original.py # 差分 vs 原始目标消融实验
│   ├── step14_cross_validation.py # 滚动起点交叉验证（稳健性）
│   ├── step15_temp_ushape.py      # 温度-电力 U 型关系量化建模
│   ├── step16_training_loss.py    # 三模型训练损失曲线对比（图6-1）
│   ├── step17_schematics.py       # 数据流程图 / 建模框架图（图2-1、图5-1）
│   └── step18_gru_residual.py     # BiGRU 残差诊断图（图6-6）
├── requirements.txt
└── README.md
```

---

## 四、技术路线

```
原始数据
   │
   ├─ Open-Meteo  ──┐
   ├─ AEP Hourly ──┤── step1~4 数据获取 / 聚合 / 合并
   └─ 节假日库   ──┘
                    │
                    ▼
          step5 EDA & 可视化（9 张图）
                    │
                    ▼
          step6 特征工程
            · 滞后特征 lag_{1,2,3,7,14,30}
            · 滑动统计 rolling_{mean,std,max,min}_{7,30}
            · 温度衍生 CDD / HDD / tavg² / temp_range
            · 周期编码 sin/cos
            · 季节 One-Hot
            · 时序划分 70 / 15 / 15
                    │
                    ▼
         ┌────────── 深度学习建模（差分预测）──────────┐
         │   step7 LSTM     step8 GRU    step9 CNN-LSTM │
         └──────────────────────────────────────────────┘
                    │
                    ▼
          step10 模型对比 + 影响因素分析
```

**关键技巧——差分预测**：直接预测 `y_t` 时模型会"抄昨天"，预测曲线滞后。本项目改为预测 **Δy = y_t − y_{t−1}**，再加回 `y_{t−1}` 还原，显著提升精度。

---

## 五、模型对比结果

测试集（2016-07 ~ 2018-08，726 个序列样本）：

| 模型 | RMSE | MAE | MAPE | R² | 参数量 | 训练时间 |
| --- | --- | --- | --- | --- | --- | --- |
| **LSTM (Diff)** | 14866.3 | 10919.1 | **3.04%** | 0.880 | 96 065 | 58.7 s |
| GRU (Diff) | 14937.1 | 11163.5 | 3.12% | 0.879 | 72 833 | 67.4 s |
| CNN-LSTM (Diff) | **14743.6** | 11192.9 | 3.13% | **0.882** | 98 369 | 84.9 s |

**结论**：三模型性能高度接近（MAPE 3.04%~3.13%、R² 约 0.88）。BiLSTM 的 MAPE 最低（3.04%），CNN-LSTM 的 RMSE 与 R² 最优，BiGRU 以最少参数取得几乎相同效果；扩充样本（1310→5053 天）后 R² 由约 0.79 提升至约 0.88，差分预测策略整体有效。

### 基线对比（相同测试区间）

| 模型 | MAPE | R² |
| --- | --- | --- |
| Persistence（昨天=今天） | 5.32% | 0.679 |
| Seasonal Naive（上周同日） | 9.03% | 0.044 |
| Linear Regression（差分特征） | **2.38%** | **0.939** |
| LSTM（本研究最优深度模型） | 3.04% | 0.880 |

> 深度模型显著优于朴素基线；但在强特征工程下，线性回归这一简单基线同样很强（甚至更优），说明本任务的可预测性主要来自**特征工程**，深度模型的价值在于自动建模、无需人工设计全部特征。

### 差分预测消融（BiLSTM）

| 策略 | MAPE | R² | 预测均值偏差 |
| --- | --- | --- | --- |
| 原始目标 | 3.92% | 0.811 | -3 923 MWh |
| 原始目标 + Bias 校正 | 3.96% | 0.805 | -5 115 MWh |
| **差分目标** | **3.04%** | **0.880** | **+55 MWh** |

> 扩充样本后原始目标的训练-测试水平偏移已减小，Bias 校正不再带来收益（甚至略降）；但差分目标仍显著优于原始目标（MAPE 3.92%→3.04%，R² 0.811→0.880，均值偏差近乎归零）。

### 递归多步未来预测（LSTM，最后 30 天）

| 方式 | MAPE | R² |
| --- | --- | --- |
| 单步（已知前一天真实值） | 2.32% | 0.813 |
| 递归 30 天（预测值回灌） | 9.15% | -0.617 |

> 递归多步误差随预测步长累积，符合时序预测规律；说明短期（1~3 天）预测可靠，长程预测需配合滚动更新或概率区间。

### 稳健性验证：4 折滚动起点交叉验证

为检验上述结论是否跨时间稳健（而非单一测试集的偶然），采用扩展窗口的滚动起点交叉验证（4 个连续年度时间折），各模型 MAPE / R² 的 **mean ± std**：

| 模型 | MAPE (mean ± std) | R² (mean ± std) |
| --- | --- | --- |
| **Linear Regression** | **2.42% ± 0.13** | **0.939 ± 0.006** |
| BiLSTM | 3.17% ± 0.15 | 0.881 ± 0.017 |
| BiGRU | 3.07% ± 0.12 | 0.887 ± 0.012 |
| CNN-LSTM | 3.04% ± 0.10 | 0.889 ± 0.017 |

> **关键结论**：线性回归在全部 4 折上均稳定优于三种深度模型（每折 MAPE 2.33%~2.61%），其优势**不是单次划分的偶然**。各模型方差都很小（std 约 0.1~0.15），结果可复现。这进一步佐证：在强特征工程下，本任务的可预测性主要由特征决定；深度模型与线性模型在精度上相近、线性模型反而更稳更省，深度学习的价值体现在自动建模与对非线性/长程依赖的适应性上。

---

## 六、影响因素分析

基于随机森林特征重要性（详见 `figures/19_feature_importance.png`、`figures/20_factor_categories.png`）：

| 类别 | 累计重要性 | 代表特征 |
| --- | --- | --- |
| **周期编码** | 56.8% | `dayofweek`, `dayofweek_sin/cos` |
| **气象因素** | 20.3% | `tmax`, `tavg`, `tmin`, `HDD` |
| **历史用电特征** | 19.2% | `lag_1`, `lag_2`, `lag_7`, `diff_lag_1` |
| **日历特征** | 2.7% | `is_workday`, `is_weekend`, `is_holiday` |
| **时间标识** | 0.9% | `day`, `year` |

EDA 关键发现（基于 5053 天数据）：

- 温度与电力消费呈典型 **"U 型曲线"**（夏季制冷 + 冬季供暖双高峰），谷值出现在 10~16°C。
- 工作日用电量比周末 **高约 9.7%**（382 478 vs 345 544 MWh）。
- 美国联邦节假日样本中，节假日用电量比非节假日 **低约 2.6%**，其独立贡献小于周内规律。
- 季节均值：冬季最高（407 130 MWh）> 夏季（387 214）> 秋季（347 908）> 春季（345 631）。

---

## 七、环境与运行

### 依赖

```bash
pip install -r requirements.txt
```

测试环境：Python 3.13 · PyTorch 2.x · macOS / Linux

### 一键运行

```bash
# 数据准备
python code/step1_get_weather.py
python code/step2_get_holiday.py
python code/step3_process_electricity.py   # 需先放入 AEP_hourly.csv
python code/step4_merge_data.py

# 分析与建模
python code/step5_eda.py
python code/step6_feature_engineering.py
python code/step7_lstm_diff.py
python code/step8_gru.py
python code/step9_cnn_lstm.py

# 综合对比 + 拓展实验
python code/step10_compare_analyze.py
python code/step11_baseline.py
python code/step12_future_forecast.py
python code/step13_diff_vs_original.py
python code/step14_cross_validation.py
python code/step15_temp_ushape.py
python code/step16_training_loss.py
python code/step17_schematics.py
python code/step18_gru_residual.py
```

> Kaggle 原始数据 `AEP_hourly.csv` 需先手动下载放入 `data/`（[数据集链接](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)）。若缺失该文件，`step3` 无法执行；仓库已附带处理好的 `final_dataset.csv`，可直接从 `step5` 开始运行。

---

## 八、可视化输出

`figures/` 目录包含当前分析与建模图表：

| 编号 | 内容 |
| --- | --- |
| 01–09 | EDA：时序、季节、周末、节假日、温度 U 型、相关性、分解、热力图、分布 |
| 11 / 13 / 14 | LSTM / GRU / CNN-LSTM 训练损失与预测曲线 |
| 15–18 | 三模型预测对比 / 指标对比 / 误差分布 / 散点对比 |
| 19–20 | 特征重要性与因素类别贡献 |
| 21 | 基线模型 vs 深度学习对比 |
| 22 | 递归多步未来预测 vs 单步预测 |
| 23 | 4 折滚动起点交叉验证 MAPE 对比 |
| 24 | 温度-电力 U 型关系量化建模（线性/二次/分段度日）|
| 25 | BiGRU 残差时序图与分布直方图（报告图6-6）|
| 26 | 三模型训练/验证损失曲线对比（报告图6-1）|
| 27 / 28 | 数据获取流程示意图 / 整体建模框架示意图（报告图2-1、图5-1）|

---

## 九、对照课设要求

| 要求项 | 完成情况 |
| --- | --- |
| 至少一个城市/区域的电力数据，含 ≥5 维信息 | ✅ 46 维建模特征（气象 + 时间 + 节假日 + 滞后 + 周期编码） |
| 不同时间段（季节、工作日/周末）趋势分析 | ✅ EDA 图 02 / 03 / 04 / 08 |
| 构建预测模型预测未来电力需求 | ✅ LSTM / GRU / CNN-LSTM + 递归多步未来预测（step12） |
| 探讨影响因素（天气、节假日等） | ✅ 特征重要性 + 因素类别分析 |
| 数据处理 | ✅ step1–4 全流程、缺失值填充、时序对齐 |
| 建模与算法应用（≥2 种深度学习） | ✅ 三种 + 基线对比（step11） |
| 模型评价与优化 | ✅ RMSE / MAE / MAPE / R² + 差分消融（step13）+ 滚动起点交叉验证（step14） |
| 结果分析与应用 | ✅ step10 综合对比 + 因素归因 |
| 报告与展示 | 见配套 PPT 与课程设计报告 |

---

## 十、数据来源声明

- **AEP Hourly Energy Consumption**: Kaggle (Rob Mulla)，PJM 公开数据
- **历史气象**: Open-Meteo Historical Weather API（CC BY 4.0），Columbus, Ohio
- **节假日**: pandas `USFederalHolidayCalendar`
