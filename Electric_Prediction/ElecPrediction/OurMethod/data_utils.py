import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

power_cols = ['global_active_power', 'global_reactive_power', 'sub_metering_1',
              'sub_metering_2', 'sub_metering_3', 'sub_metering_remainder',
              'voltage', 'global_intensity']

feature_cols = ['global_reactive_power', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3',
                    'sub_metering_remainder', 'voltage', 'global_intensity',
                    'rr', 'nbjrr1', 'nbjrr5', 'nbjrr10', 'nbjbrou',
                    'doy_sin', 'doy_cos', 'year_norm']

label_col = 'global_active_power'

def preprocess(df, fit_scaler=False, feature_scaler=None, label_scaler=None):
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['day_of_year'] = df['DateTime'].dt.dayofyear
    df['year'] = df['DateTime'].dt.year
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    features = df[feature_cols].astype(np.float32)
    labels = df[label_col].astype(np.float32).values.reshape(-1, 1)

    if fit_scaler:
        feature_scaler, label_scaler = StandardScaler(), StandardScaler()
        features_scaled = feature_scaler.fit_transform(features)
        labels_scaled = label_scaler.fit_transform(labels)
    else:
        features_scaled = feature_scaler.transform(features)
        labels_scaled = label_scaler.transform(labels)

    return features_scaled, labels_scaled.ravel(), feature_scaler, label_scaler

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len, pred_len):
        self.X, self.y = X, y
        self.seq_len, self.pred_len = seq_len, pred_len

    def __len__(self):
        return len(self.X) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_seq = self.y[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.tensor(x_seq), torch.tensor(y_seq)