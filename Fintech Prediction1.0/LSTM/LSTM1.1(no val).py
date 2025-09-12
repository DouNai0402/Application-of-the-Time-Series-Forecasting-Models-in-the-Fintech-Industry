import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# read data
SPY_DF = pd.read_csv("../Dataset/processed_data/GLD_processed.csv", index_col='Date', parse_dates=True)

# set features and target
FEATURES = ['Open','High','Low','Close','Volume']
for col in FEATURES:
    SPY_DF[col] = np.log(SPY_DF[col] + 1e-8)
TARGET = 'Close'

# use slide window to create sequence
def create_sequence(df, feature_cols, target_col,seq_length):
    features = df[feature_cols].values
    # reshape the target(row vectors to column vectors)
    target = df[target_col].values.reshape(-1,1)

    X, y = [], []
    for  i  in range(len(features) - seq_length):   #len(features) is the number of data
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

# model training and prediction
SEQ_LEN = 60
X, y = create_sequence(SPY_DF, FEATURES, TARGET, SEQ_LEN)

# split test set and validation set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.15, shuffle=False
)

feature_scaler = StandardScaler()
target_scaler= StandardScaler()

# fit_transform train set and transform val set and test set
def scale_X(X, scaler):
    shape = X.shape     # X.shape is (batch_size, seq_len, feature_num)
    X_reshaped = X.reshape(-1, shape[-1])   # StandardScaler only accept 2 dimension
    X_scaled = scaler.fit_transform(X_reshaped).reshape(shape)
    return X_scaled

def transform_X(X,scaler):
    shape = X.shape
    X_reshaped = X.reshape(-1, shape[-1])
    X_scaled = scaler.transform(X_reshaped).reshape(shape)
    return X_scaled

X_train_scaled = scale_X(X_train, feature_scaler)
X_test_scaled = transform_X(X_test, feature_scaler)

y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)


# convert vector to tensor
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

# create LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(LSTMModel,self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()   #set non-linear activations
        self.fc = nn.Linear(hidden_dim, 1)  #set linear function and predict length

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]     #get the hidden status of the last step
        last_time_step = self.dropout(last_time_step)
        relu_out = self.relu(last_time_step)
        prediction = self.fc(relu_out)
        return prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# create Dataset and DataLoader to load data
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False, pin_memory=True)

# initialize model
model = LSTMModel(input_dim=len(FEATURES), hidden_dim=256, num_layers=2, dropout_rate=0.1).to(device)
loss_fn = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train model
EPOCHS = 200
for epoch in range(EPOCHS):
    model.train()
    total_loss_t = 0
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        output = model(X_train_batch)
        loss = loss_fn(output, y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_t += loss.item() * X_train_batch.size(0)
    avg_loss_t = total_loss_t / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS}: Train Loss = {avg_loss_t:.6f}")

# evaluation and print
model.eval()
all_predictions, all_targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predictions = model(X_batch)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

predictions = np.vstack(all_predictions)
actual = np.vstack(all_targets)

# inverse_transform
predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
actual      = target_scaler.inverse_transform(actual.reshape(-1, 1)).reshape(actual.shape)
predictions = np.exp(predictions)
actual = np.exp(actual)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

rmse = np.sqrt(mean_squared_error(actual, predictions))
mae = mean_absolute_error(actual, predictions)
r2 = r2_score(actual, predictions)
mape = mean_absolute_percentage_error(actual, predictions)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

output_dir = "../Results"
os.makedirs(output_dir, exist_ok=True)

metrics_path = os.path.join(output_dir, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"R2: {r2:.4f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")

print(f"Metrics saved to {metrics_path}")

plt.figure(figsize=(14,6))
plt.plot(actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title("AMZN Stock Price Prediction with LSTM (PyTorch)")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plot_path = os.path.join(output_dir, "prediction_plot.png")
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to {plot_path}")