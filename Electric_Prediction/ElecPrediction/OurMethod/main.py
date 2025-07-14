from data_utils import preprocess, TimeSeriesDataset, power_cols, label_col
from models import DecomposedTransformer
from train_eval import train_model, evaluate_model
from plot_utils import plot_components

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

seq_len = 90
pred_len = 90 #or 365
batch_size = 32
d_model = 128 #64 for long-term
nhead = 4
num_layers = 4 # 2 for long-term
epochs = 80

if __name__ == '__main__':
    df_train = pd.read_csv('../Processed_train.csv')
    df_test = pd.read_csv('../Processed_test.csv')

    df_test_all = df_test.copy()

    df_train['DateTime'] = pd.to_datetime(df_train['DateTime'])
    df_test['DateTime'] = pd.to_datetime(df_test['DateTime'], format='%Y/%m/%d')

    df_train.replace('', pd.NA, inplace=True)
    df_test.replace('', pd.NA, inplace=True)

    invalid_rows = df_train[power_cols].isna().all(axis=1) | (df_train[power_cols].sum(axis=1) == 0)
    df_train = df_train[~invalid_rows].reset_index(drop=True)

    mask_zero = df_test[power_cols].isna().all(axis=1) | (df_test[power_cols].sum(axis=1) == 0)
    zero_indices = df_test[mask_zero].index
    df_test = df_test[~mask_zero].reset_index(drop=True)

    X_train, y_train, feature_scaler, label_scaler = preprocess(df_train, fit_scaler=True)
    X_test, y_test, _, _ = preprocess(df_test, fit_scaler=False, feature_scaler=feature_scaler, label_scaler=label_scaler)

    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_len, pred_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_short = RSDTransformer(input_dim=X_train.shape[1], d_model=d_model, nhead=nhead,
    #                        num_layers=num_layers, seq_len=seq_len, pred_len=pred_len).to(device)

    model = DecomposedTransformer(
        input_dim=X_train.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        pred_len=pred_len,
        kernel_size=seq_len
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    criterion = nn.MSELoss()

    train_model(model, train_loader, criterion, optimizer, scheduler=scheduler, epochs=epochs)
    # torch.save(model.state_dict(), "final_model_365.pth")
    torch.save(model.state_dict(), "final_model_90.pth") #

    preds_real, targets_real = evaluate_model(model, test_loader, label_scaler)

    model.eval()
    with torch.no_grad():
        x_input = torch.tensor(X_test[-seq_len:]).unsqueeze(0).to(device)  # [1, seq_len, input_dim]
        pred_scaled, trend_scaled, seasonal_scaled, alpha = model(x_input)
        pred_real = label_scaler.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
        trend_real = label_scaler.inverse_transform(trend_scaled.cpu().numpy().reshape(-1, 1)).flatten()
        seasonal_real = label_scaler.inverse_transform(seasonal_scaled.cpu().numpy().reshape(-1, 1)).flatten()

    truth_real = df_test_all[label_col].values[-pred_len:]

    plot_components(truth_real, pred_real, trend_real, seasonal_real)