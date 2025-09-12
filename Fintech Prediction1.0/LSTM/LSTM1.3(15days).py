import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TICKERS    = ["SPY","BND","GLD","VIOO"]
CSV_TEMPL  = "../Dataset/processed_data/{sym}_processed.csv"
FEATURES   = ['Open','High','Low','Close','Volume']
TARGET     = 'Close'
SEQ_LEN    = 96
HORIZON    = 15
BATCH_SIZE = 64
EPOCHS     = 100
LR         = 1e-3
HIDDEN     = 256
LAYERS     = 2
DROPOUT    = 0.1
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

def create_sequence(df, feature_cols, target_col, seq_length, horizon):
    """
    X: [N, seq_len, features]
    y: [N, horizon]
    """
    features = df[feature_cols].values
    target   = df[target_col].values.reshape(-1, 1)
    X, y = [], []

    for i in range(len(features) - seq_length - horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length : i + seq_length + horizon, 0])
    return np.array(X), np.array(y)  # X:[N, L, F], y:[N, H]

def scale_X_fit(X, scaler: StandardScaler):
    shape = X.shape  # (N, L, F)
    X_reshaped = X.reshape(-1, shape[-1])  # (N*L, F)
    X_scaled = scaler.fit_transform(X_reshaped).reshape(shape)
    return X_scaled

def scale_X_transform(X, scaler: StandardScaler):
    shape = X.shape
    X_reshaped = X.reshape(-1, shape[-1])
    X_scaled = scaler.transform(X_reshaped).reshape(shape)
    return X_scaled

def mape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate, horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)         # [B, L, H]
        last_time_step = lstm_out[:, -1, :]# [B, H]
        h = self.dropout(self.relu(last_time_step))
        pred = self.fc(h)                  # [B, horizon]
        return pred

# -----------------------
# Core: train & evaluate one ticker
# -----------------------
def run_one_ticker(sym: str, csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist.")

    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True).copy()

    df = df.copy()
    for col in FEATURES:
        df[col] = np.log(df[col] + 1e-8)

    X, y = create_sequence(df, FEATURES, TARGET, SEQ_LEN, HORIZON)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )

    feat_scaler = StandardScaler()
    targ_scaler = StandardScaler()

    X_train_s = scale_X_fit(X_train, feat_scaler)
    X_test_s  = scale_X_transform(X_test, feat_scaler)

    y_train_s = targ_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_test_s  = targ_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    # 转 tensor
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_s,  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),
                              batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = LSTMModel(
        input_dim=len(FEATURES),
        hidden_dim=HIDDEN,
        num_layers=LAYERS,
        dropout_rate=DROPOUT,
        horizon=HORIZON
    ).to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)          # xb:[B,L,F], yb:[B,H]
            pred = model(xb)                               # [B,H]
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"[{sym}] Epoch {epoch+1}/{EPOCHS} TrainLoss={total/len(train_loader.dataset):.6f}")

    model.eval()
    all_preds, all_targs = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            pr = model(xb).cpu().numpy()  # [B,H]
            tg = yb.numpy()               # [B,H]
            all_preds.append(pr); all_targs.append(tg)

    preds = np.vstack(all_preds)   # [N,H]
    targs = np.vstack(all_targs)   # [N,H]

    preds_real = targ_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    targs_real = targ_scaler.inverse_transform(targs.reshape(-1, 1)).reshape(targs.shape)
    preds_real = np.exp(preds_real)
    targs_real = np.exp(targs_real)

    preds_f = preds_real.reshape(-1)
    targs_f = targs_real.reshape(-1)

    rmse = np.sqrt(mean_squared_error(targs_f, preds_f))
    mae  = mean_absolute_error(targs_f, preds_f)
    mp   = mape(targs_f, preds_f)

    metrics = {"symbol": sym, "RMSE": rmse, "MAE": mae, "MAPE(%)": mp}
    print(metrics)
    return metrics

# -----------------------
# Run all tickers
# -----------------------
all_metrics = []
for sym in TICKERS:
    csv_path = CSV_TEMPL.format(sym=sym)
    print(f"\n=== Running {sym} ===")
    metrics = run_one_ticker(sym, csv_path)
    all_metrics.append(metrics)

print("\nSummary:")
df_metrics = pd.DataFrame(all_metrics).set_index("symbol")
df_metrics = df_metrics[["RMSE","MAE","MAPE(%)"]]
print(df_metrics.round(4).to_string)
