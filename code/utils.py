"""
utils.py
项目公共模块：随机种子、特征构造、序列化、评价指标、模型结构。
被 step7~step13 共同复用，避免重复代码、保证特征口径一致。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

TARGET = 'electricity_total'
SEQ_LEN = 28
SEED = 42

# 建模时不参与输入的列
DROP_COLS = ['date', 'electricity_mean', 'electricity_max', 'electricity_min',
             'electricity_total', 'target_diff']


def set_seed(seed: int = SEED):
    """统一设定随机种子，保证结果可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """在已有原始列的 df 上构造差分预测所需的全部特征（不做 dropna）。

    被 build_diff_frame 与递归预测(step12)共用，保证特征口径完全一致。
    """
    df = df.sort_values('date').reset_index(drop=True).copy()

    # 差分目标
    df['target_diff'] = df[TARGET].diff()

    # 滞后特征
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'lag_{lag}'] = df[TARGET].shift(lag)

    # 滑动统计（基于 t-1 及之前，避免数据泄露）
    for window in [7, 30]:
        df[f'rolling_mean_{window}'] = df[TARGET].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df[TARGET].shift(1).rolling(window).std()
        df[f'rolling_max_{window}'] = df[TARGET].shift(1).rolling(window).max()
        df[f'rolling_min_{window}'] = df[TARGET].shift(1).rolling(window).min()

    # 差分滞后
    for lag in [1, 7]:
        df[f'diff_lag_{lag}'] = df['target_diff'].shift(lag)

    # 温度衍生特征
    df['CDD'] = (df['tavg'] - 18).clip(lower=0)
    df['HDD'] = (18 - df['tavg']).clip(lower=0)
    df['tavg_sq'] = df['tavg'] ** 2
    df['temp_range'] = df['tmax'] - df['tmin']

    # 季节 One-Hot（固定 4 列顺序，保证特征对齐）
    for s in ['autumn', 'spring', 'summer', 'winter']:
        df[f'season_{s}'] = (df['season'] == s).astype(int)
    df = df.drop(columns=['season'])
    return df


def build_diff_frame(path: str = 'data/final_dataset.csv'):
    """读取最终数据集并构造差分预测所需的全部特征。

    返回 (df_clean, feature_cols)。特征口径与 step7 完全一致。
    """
    df = pd.read_csv(path, parse_dates=['date'])
    df = add_diff_features(df)
    df_clean = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df_clean.columns if c not in DROP_COLS]
    return df_clean, feature_cols


def time_split(df_clean: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    """时间序列按时间顺序切分 train/val/test。"""
    n = len(df_clean)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (df_clean.iloc[:train_end],
            df_clean.iloc[train_end:val_end],
            df_clean.iloc[val_end:])


def prepare_diff_data(path: str = 'data/final_dataset.csv'):
    """完整的差分数据准备流程，返回训练所需的全部数组与 scaler。"""
    df_clean, feature_cols = build_diff_frame(path)
    train, val, test = time_split(df_clean)

    def xy(part):
        return part[feature_cols].values, part[['target_diff']].values

    X_train, y_train_diff = xy(train)
    X_val, y_val_diff = xy(val)
    X_test, y_test_diff = xy(test)

    prev_train = train[TARGET].shift(1).fillna(train[TARGET].iloc[0]).values
    prev_val = val[TARGET].shift(1).fillna(val[TARGET].iloc[0]).values
    prev_test = test[TARGET].shift(1).fillna(test[TARGET].iloc[0]).values

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train_diff)
    y_val_s = scaler_y.transform(y_val_diff)
    y_test_s = scaler_y.transform(y_test_diff)

    return {
        'feature_cols': feature_cols,
        'df_clean': df_clean,
        'train': train, 'val': val, 'test': test,
        'X_train_s': X_train_s, 'y_train_s': y_train_s,
        'X_val_s': X_val_s, 'y_val_s': y_val_s,
        'X_test_s': X_test_s, 'y_test_s': y_test_s,
        'prev_train': prev_train, 'prev_val': prev_val, 'prev_test': prev_test,
        'truth_train': train[TARGET].values,
        'truth_val': val[TARGET].values,
        'truth_test': test[TARGET].values,
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
    }


def create_sequences_with_prev(X, y, prev, truth, seq_len=SEQ_LEN):
    """滑动窗口序列化，同时保留每个样本的“前一天真实值”用于差分还原。"""
    Xs, ys, prevs, truths = [], [], [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
        prevs.append(prev[i + seq_len])
        truths.append(truth[i + seq_len])
    return np.array(Xs), np.array(ys), np.array(prevs), np.array(truths)


def make_loader(X, y, batch_size=32, shuffle=False):
    return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
                      batch_size=batch_size, shuffle=shuffle)


def compute_metrics(preds, truths):
    """统一计算 RMSE / MAE / MAPE / R²。"""
    preds = np.asarray(preds, dtype=float)
    truths = np.asarray(truths, dtype=float)
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    mae = float(np.mean(np.abs(preds - truths)))
    mape = float(np.mean(np.abs((preds - truths) / truths)) * 100)
    r2 = float(1 - np.sum((truths - preds) ** 2) / np.sum((truths - truths.mean()) ** 2))
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


# ============================================================
# 模型结构
# ============================================================
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=48, num_layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=48, num_layers=2, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0,
                          bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_channels=32, hidden_size=48,
                 num_layers=2, dropout=0.25, kernel_size=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
