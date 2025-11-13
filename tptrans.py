import os
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from dataloader import load_parquet, preprocess_data, SlidingWindowDataset


class WeightedStepMSELoss(nn.Module):
    """
    MSE loss with increasing weights for later steps in a sequence.
    
    mode: 'linear' or 'exponential' weighting. Default: Equal weighting (None).
    start_weight: Lowest assigned weight
    end_weight: Highest assigned weight
    """
    def __init__(self, mode="", start_weight=100.0, end_weight=1000.0):
        super().__init__()
        self.mode = mode
        self.start_weight = start_weight
        self.end_weight = end_weight

    def forward(self, y_pred, y_true):
        # y_pred, y_true: (batch_size, seq_len, features)
        batch_size, seq_len, n_features = y_pred.shape

        # Create weights along the sequence dimension
        if self.mode.startswith('lin'):
            weights = torch.linspace(self.start_weight, self.end_weight, steps=seq_len, device=y_pred.device)
        elif self.mode.startswith('exp'):
            weights = torch.exp(torch.linspace(self.start_weight, self.end_weight, steps=seq_len, device=y_pred.device))
        else:
            weights = torch.ones(seq_len, device=y_pred.device)

        # Reshape for broadcasting: (1, seq_len, 1)
        weights = weights.view(1, seq_len, 1)

        # Compute squared error
        se = (y_pred - y_true) ** 2

        # Apply weights per step (broadcast over batch and features)
        weighted_se = se * weights

        # Mean over batch, sequence, and features
        loss = weighted_se.mean()
        return loss

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
    def __init__(
        self,
        input_dim=6,
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        pred_len=5,
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, d_model, kernel_size=5, padding=1)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model)
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        self.decoder_input_proj = nn.Linear(2, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, 4 * d_model)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.output_head = nn.Linear(d_model, 2)  # lat, lon
        self.pred_len = pred_len

    def forward(self, src, tgt_start):
        # src: (batch, src_len, input_dim)
        # tgt_start: start token or last known position (batch, 1, input_dim)
        src = src.permute(1, 0, 2)
        src = self.conv(src.permute(1, 2, 0)).permute(2, 0, 1)
        src = self.pos_encoder(src)
        memory = self.encoder(src)

        # initial decoder input: project last known position
        tgt = self.decoder_input_proj(tgt_start.permute(1, 0, 2))  # (1, batch, d_model)
        tgt = self.pos_decoder(tgt)

        preds = []
        for _ in range(self.pred_len):
            out = self.decoder(tgt, memory)  # (seq_len, batch, d_model)
            step = self.output_head(out[-1])  # (batch, 2)
            preds.append(step)

            # project predicted step to embedding space for next timestep
            step_emb = self.decoder_input_proj(step).unsqueeze(0)  # (1, batch, d_model)
            step_emb = self.pos_decoder(step_emb)  # add positional encoding

            # append to tgt sequence
            tgt = torch.cat([tgt, step_emb], dim=0)  # now all dims match
        return torch.stack(preds, dim=1)  # (batch, pred_len, 2)

def _train(
    model,
    dataset,
    device,
    batch_size=32,
    lr=1e-5,
    epochs=100,
    val_split=0.2,
    test_split=0.1,
    save_model=False,
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
    # criterion = WeightedStepMSELoss(mode="lin")
    # criterion = nn.HuberLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # simplest: use the last observed position as start token
            tgt_start = x[:, -1:, :2]  # grab lat/lon
            optimizer.zero_grad()
            pred = model(x, tgt_start)
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
                tgt_start = x[:, -1:, 2:4]  # grab lat/lon
                pred = model(x, tgt_start)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= val_size

        if save_model and epoch % 25 == 0 and epoch > 25:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            torch.save(
                model.state_dict(),
                os.path.join("checkpoints", f"{save_model}_{str(epoch)}.pth"),
            )

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}",
            flush=True,
        )

    return model, train_loader, val_loader, test_loader


def _evaluate(model, test_loader, device):
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            tgt_start = x[:, -1:, 2:4]  # (batch, 1, 2) lat/lon
            pred = model(x, tgt_start)
            loss = criterion(pred, y)
            test_loss += loss.item() * x.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}", flush=True)
    return test_loss


def train_model_from_dataset(k=100, epochs=100, save_model=""):
    df = load_parquet("aisdk-2025-02-27", k=k)
    df = preprocess_data(df)

    dataset = SlidingWindowDataset(
        df,
        max_diff_per_sequence_minutes=15,
        window_size_minutes=420,
        pred_size_minutes=120,
        stride=15,
    )
    print(f"Total dataset size: {len(dataset)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...", flush=True)
    model = TPTrans(pred_len=dataset[0][1].shape[0]).to(device)

    trained_model, train_loader, val_loader, test_loader = _train(
        model,
        dataset,
        device,
        batch_size=64,
        lr=1e-5,
        epochs=epochs,
        save_model=save_model,
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
    model = train_model_from_dataset(
        k=1000, epochs=200, save_model="tptrans_delta_lin_newNewParams"
    )
