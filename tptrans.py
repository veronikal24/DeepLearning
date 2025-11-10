import os
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from dataloader import load_parquet, preprocess_data, SlidingWindowDataset


# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return x


# --- TPtrans-like Model ---
class TPTrans(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=8, num_layers=4, pred_len=5):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.decoder_lat = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, pred_len)
        )
        self.decoder_lon = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, pred_len)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = self.conv(x)  # (batch, d_model, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)

        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model) for transformer
        encoded = self.transformer_encoder(x)
        encoded = encoded.mean(dim=0)  # (batch, d_model)

        lat_pred = self.decoder_lat(encoded)  # (batch, pred_len)
        lon_pred = self.decoder_lon(encoded)  # (batch, pred_len)

        y_pred = torch.stack([lat_pred, lon_pred], dim=-1)  # (batch, pred_len, 2)
        return y_pred


def _train(
    model,
    dataset,
    device,
    batch_size=32,
    lr=1e-5,
    epochs=100,
    val_split=0.2,
    test_split=0.1,
):
    # Calculate sizes for splits
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / train_size

        # Optional: compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= val_size

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}",
            flush=True,
        )
        with open(os.path.join("checkpoints", "current_job.log"), "a") as f:
            f.write(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}\n"
            )

    return model, train_loader, val_loader, test_loader


def _evaluate(model, test_loader, device):
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            test_loss += loss.item() * x.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}", flush=True)
    return test_loss


def train_model_from_dataset(k=100, epochs=100, save_model=""):
    df = load_parquet("dataset", k=k)
    df = preprocess_data(df)

    dataset = SlidingWindowDataset(
        df,
        max_diff_per_sequence_minutes=15,
        window_size_minutes=240,
        pred_size_minutes=60,
        stride=60,
    )
    print(f"Total dataset size: {len(dataset)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...", flush=True)
    model = TPTrans(pred_len=dataset[0][1].shape[0]).to(device)

    trained_model, train_loader, val_loader, test_loader = _train(
        model, dataset, device, batch_size=32, lr=1e-5, epochs=epochs
    )

    _evaluate(trained_model, test_loader, device)

    if save_model:
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        torch.save(
            trained_model.state_dict(), os.path.join("checkpoints", f"{save_model}.pth")
        )
        # to load later:
        # model = TPTrans()
        # model.load_state_dict(torch.load(filepath))
        # model.eval()

    return trained_model


if __name__ == "__main__":
    with open(os.path.join("checkpoints", "current_job.log"), "w") as f:
        f.write("\nStarting training...\n\n")
    model = train_model_from_dataset(
        k=500, epochs=1000, save_model="tptrans_k500_e1000"
    )
