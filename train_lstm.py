# train_lstm.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# --------------------------
# Device (GPU if available)
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# LSTM Dataset
# --------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------------------------
# LSTM Model
# --------------------------
class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # last timestep
        return self.fc(output)

# --------------------------
# Prepare Data
# --------------------------
def create_sequences(data, target, seq_len=20):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(target[i])
    return np.array(X), np.array(y)

# --------------------------
# Main training function
# --------------------------
def train_lstm(dataset_csv="TSLA_dataset.csv"):

    print("\n📌 Loading dataset...")
    df = pd.read_csv(dataset_csv)

    # Features to use (drop target & date)
    feature_cols = [c for c in df.columns if c not in ["target", "date"]]

    X_raw = df[feature_cols].values
    y_raw = df["target"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Create sequences
    seq_len = 20
    X, y = create_sequences(X_scaled, y_raw, seq_len=seq_len)

    # Train-test split (time-based)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Model
    num_features = X.shape[2]
    model = LSTMModel(num_features).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n🚀 Training LSTM model...\n")
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(train_loader):.4f}")

    # Save model + scaler
    torch.save(model.state_dict(), "lstm_model.pth")
    joblib.dump(scaler, "scaler.pkl")

    print("\n✅ Model saved as lstm_model.pth")
    print("✅ Scaler saved as scaler.pkl")

    return model

# --------------------------
# Run directly
# --------------------------
if __name__ == "__main__":
    train_lstm()
