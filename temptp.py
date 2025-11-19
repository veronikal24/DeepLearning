import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import random_split, DataLoader
from dataloader import (
    load_parquet,
    preprocess_data,
    SlidingWindowDatasetTemporal,
    SAMPLE_INTERVAL_MIN,
)
from utils import deltas_to_coords
from informer_original_paper import (
    ProbAttention,
    EncoderLayer,
    AttentionLayer,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len].unsqueeze(1).to(x.device).to(x.dtype)


class TemporalEncoding(nn.Module):
    def __init__(self, time_dim, d_model, num_freqs=4):
        super().__init__()

        if time_dim != 6:
            print("Currently only 6 time dims (hour, day, month) implemented")
            raise Exception

        self.time_dim = time_dim
        self.d_model = d_model
        self.num_freqs = num_freqs

        # projections per component
        # each gets (2 * num_freqs * 2) dims after expansion
        expanded_dim = 2 * num_freqs * 2

        self.proj_hour = nn.Linear(expanded_dim, d_model)
        self.proj_dow = nn.Linear(expanded_dim, d_model)
        self.proj_month = nn.Linear(expanded_dim, d_model)

        # bias parameters for projections
        self.comp_bias = nn.Parameter(torch.zeros(3, d_model))

    def _expand(self, x):
        # Expand input with sin(k*x) and cos(k*x) for better high-freq awareness?
        freqs = (
            torch.arange(self.num_freqs, device=x.device, dtype=x.dtype).view(
                1, 1, 1, self.num_freqs
            )
            + 1.0
        )

        # (batch, seq, 2, 1)
        x = x.unsqueeze(-1)

        sin = torch.sin(freqs * x)
        cos = torch.cos(freqs * x)

        # (batch, seq, 2, num_freqs, 2)
        enc = torch.stack([sin, cos], dim=-1)

        # flatten to (batch, seq, expanded_dim)
        return enc.flatten(-3)

    def forward(self, t):
        hour = t[:, :, 0:2]
        day = t[:, :, 2:4]
        month = t[:, :, 4:6]

        # expand each dimension
        hour_exp = self._expand(hour)
        day_exp = self._expand(day)
        month_exp = self._expand(month)

        # project individually
        h = self.proj_hour(hour_exp)
        d = self.proj_dow(day_exp)
        m = self.proj_month(month_exp)

        # combine projections and learned biases
        out = h + self.comp_bias[0] + d + self.comp_bias[1] + m + self.comp_bias[2]

        return out


class TempTPT(nn.Module):
    def __init__(
        self,
        input_dim=12,
        time_dim=6,
        d_model=256,
        nhead=8,
        num_encoder_layers=8,
        pred_len=6,
        conv_kernel=3,
        conv_padding=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.pos_dims = self.input_dim - self.time_dim

        # project 6 inputs to d_model
        self.input_proj = nn.Linear(self.pos_dims, d_model)
        self.temp_encoder = TemporalEncoding(time_dim, d_model)

        # positional encoding as first block
        self.pos_encoder = PositionalEncoding(d_model, max_len=2048)
        # convolution for short-term feature extraction
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel, padding=conv_padding
        )

        # Transformer encoder for global extraction
        ffn_dim = 4 * d_model
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ffn_dim)
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        # decoder positional encoding
        self.pos_decoder = PositionalEncoding(d_model, max_len=2048)

        # two-channel sequential linear layer decoding for prediction
        self.decoder_delta_lat = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 1)
        )
        self.decoder_delta_lon = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 1)
        )

    def forward(self, src, pred_len=None):
        if pred_len is None:
            pred_len = self.pred_len

        batch, src_len, d_model = src.shape

        # Split 6 positional + 6 temporal dims
        src_main = src[:, :, : self.pos_dims]  # (batch, src_len, 6)
        src_time = src[:, :, self.pos_dims :]  # (batch, seq_len, time_dim)

        # Encode each separately
        x_main = self.input_proj(src_main)  # (batch, src_len, d_model)
        x_temp = self.temp_encoder(src_time)  # (batch, seq_len, d_model)

        # Combine
        x = x_main + x_temp  # (batch, src_len, d_model)

        # positional encoding
        x = x.permute(1, 0, 2)  # (src_len, batch, d_model)
        x = self.pos_encoder(x)

        # convolution
        x_conv = x.permute(1, 2, 0)  # (batch, d_model, src_len)
        x_conv = self.conv(x_conv)
        memory = x_conv.permute(2, 0, 1)  # (src_len, batch, d_model)

        # transformer encoder
        memory = self.encoder(memory)
        # global context (mean pooling)
        context = memory.mean(dim=0)

        # init decoder positional encodings for future steps with zeros
        zeros_for_pos = torch.zeros(
            pred_len, batch, self.d_model, device=src.device, dtype=memory.dtype
        )
        tgt_pos = self.pos_decoder(zeros_for_pos)

        # Construct time dimensions of future predictions
        if self.time_dim == 6:
            last_hour = src_time[:, -1, 0] * (24 / (2 * np.pi))
            last_dow = src_time[:, -1, 2] * (7 / (2 * np.pi))
            last_month = src_time[:, -1, 4] * (12 / (2 * np.pi))
            steps = torch.arange(pred_len, device=src.device).unsqueeze(0)
            fh = (last_hour.unsqueeze(1) + (steps * SAMPLE_INTERVAL_MIN / 60)) % 24
            fd = (last_dow.unsqueeze(1) + (steps / 24)) % 7
            fm = (last_month.unsqueeze(1) + (steps / 720)) % 12
            future_temp = torch.stack(
                [
                    torch.sin(2 * np.pi * fh / 24),
                    torch.cos(2 * np.pi * fh / 24),
                    torch.sin(2 * np.pi * fd / 7),
                    torch.cos(2 * np.pi * fd / 7),
                    torch.sin(2 * np.pi * fm / 12),
                    torch.cos(2 * np.pi * fm / 12),
                ],
                dim=-1,
            )
        else:
            print(
                "Only 6 time dimensions (sin/cos of hour, day_of_week, month) allowed atm."
            )
        # project to d_model
        future_temp = self.temp_encoder(future_temp)  # (batch, pred_len, d_model)
        future_temp = future_temp.permute(1, 0, 2)  # (pred_len, batch, d_model)

        # combine context + positional encoding
        context_expanded = context.unsqueeze(0).expand(pred_len, -1, -1)
        dec_in = context_expanded + tgt_pos + future_temp

        # flatten for FC layers
        dec_flat = dec_in.reshape(pred_len * batch, self.d_model)

        # predict deltas
        delta_lat_flat = self.decoder_delta_lat(dec_flat)
        delta_lon_flat = self.decoder_delta_lon(dec_flat)

        delta_lat = delta_lat_flat.view(pred_len, batch)
        delta_lon = delta_lon_flat.view(pred_len, batch)

        # from (pred_len, batch) to (batch, pred_len)
        delta_lat = delta_lat.permute(1, 0)
        delta_lon = delta_lon.permute(1, 0)

        # combine lat and lon channels to one tensor
        preds = torch.stack([delta_lat, delta_lon], dim=2)

        return preds


class TempTPI(nn.Module):
    def __init__(
        self,
        input_dim=12,
        time_dim=6,
        d_model=256,
        nhead=8,
        num_encoder_layers=8,
        pred_len=6,
        conv_kernel=3,
        conv_padding=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.pos_dims = self.input_dim - self.time_dim

        # project 6 inputs to d_model
        self.input_proj = nn.Linear(self.pos_dims, d_model)
        self.temp_encoder = TemporalEncoding(time_dim, d_model)

        # positional encoding as first block
        self.pos_encoder = PositionalEncoding(d_model, max_len=2048)
        # convolution for short-term feature extraction
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel, padding=conv_padding
        )

        ffn_dim = 4 * d_model
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            5,
                            attention_dropout=0.1,
                            output_attention=False,
                        ),
                        d_model,
                        nhead,
                        mix=False,
                    ),
                    d_model=d_model,
                    d_ff=ffn_dim,
                    dropout=0.1,
                    activation="gelu",
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # decoder positional encoding
        self.pos_decoder = PositionalEncoding(d_model, max_len=2048)

        # two-channel sequential linear layer decoding for prediction
        self.decoder_delta_lat = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 1)
        )
        self.decoder_delta_lon = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 1)
        )

    def forward(self, src, pred_len=None):
        if pred_len is None:
            pred_len = self.pred_len

        batch, src_len, d_model = src.shape

        # Split 6 positional + 6 temporal dims
        src_main = src[:, :, : self.pos_dims]  # (batch, src_len, 6)
        src_time = src[:, :, self.pos_dims :]  # (batch, seq_len, time_dim)

        # Encode each separately
        x_main = self.input_proj(src_main)  # (batch, src_len, d_model)
        x_temp = self.temp_encoder(src_time)  # (batch, seq_len, d_model)

        # Combine
        x = x_main + x_temp  # (batch, src_len, d_model)

        # positional encoding
        x = x.permute(1, 0, 2)  # (src_len, batch, d_model)
        x = self.pos_encoder(x)

        # convolution
        x_conv = x.permute(1, 2, 0)  # (batch, d_model, src_len)
        x_conv = self.conv(x_conv)
        memory = x_conv.permute(2, 0, 1)  # (src_len, batch, d_model)

        # transformer encoder
        for layer in self.encoder_layers:
            memory, _ = layer(memory, attn_mask=None)
        # global context (mean pooling)
        context = memory.mean(dim=0)

        # init decoder positional encodings for future steps with zeros
        zeros_for_pos = torch.zeros(
            pred_len, batch, self.d_model, device=src.device, dtype=memory.dtype
        )
        tgt_pos = self.pos_decoder(zeros_for_pos)

        # Construct time dimensions of future predictions
        if self.time_dim == 6:
            last_hour = src_time[:, -1, 0] * (24 / (2 * np.pi))
            last_dow = src_time[:, -1, 2] * (7 / (2 * np.pi))
            last_month = src_time[:, -1, 4] * (12 / (2 * np.pi))
            steps = torch.arange(pred_len, device=src.device).unsqueeze(0)
            fh = (last_hour.unsqueeze(1) + (steps * SAMPLE_INTERVAL_MIN / 60)) % 24
            fd = (last_dow.unsqueeze(1) + (steps / 24)) % 7
            fm = (last_month.unsqueeze(1) + (steps / 720)) % 12
            future_temp = torch.stack(
                [
                    torch.sin(2 * np.pi * fh / 24),
                    torch.cos(2 * np.pi * fh / 24),
                    torch.sin(2 * np.pi * fd / 7),
                    torch.cos(2 * np.pi * fd / 7),
                    torch.sin(2 * np.pi * fm / 12),
                    torch.cos(2 * np.pi * fm / 12),
                ],
                dim=-1,
            )
        else:
            print(
                "Only 6 time dimensions (sin/cos of hour, day_of_week, month) allowed atm."
            )
        # project to d_model
        future_temp = self.temp_encoder(future_temp)  # (batch, pred_len, d_model)
        future_temp = future_temp.permute(1, 0, 2)  # (pred_len, batch, d_model)

        # combine context + positional encoding
        context_expanded = context.unsqueeze(0).expand(pred_len, -1, -1)
        dec_in = context_expanded + tgt_pos + future_temp

        # flatten for FC layers
        dec_flat = dec_in.reshape(pred_len * batch, self.d_model)

        # predict deltas
        delta_lat_flat = self.decoder_delta_lat(dec_flat)
        delta_lon_flat = self.decoder_delta_lon(dec_flat)

        delta_lat = delta_lat_flat.view(pred_len, batch)
        delta_lon = delta_lon_flat.view(pred_len, batch)

        # from (pred_len, batch) to (batch, pred_len)
        delta_lat = delta_lat.permute(1, 0)
        delta_lon = delta_lon.permute(1, 0)

        # combine lat and lon channels to one tensor
        preds = torch.stack([delta_lat, delta_lon], dim=2)

        return preds


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
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # criterion = WeightedStepMSELoss(mode="lin")
    # criterion = nn.HuberLoss()
    # criterion = PenalizedCoordLoss(nn.MSELoss())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(deltas_to_coords(x, pred), deltas_to_coords(x, y))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(deltas_to_coords(x, pred), deltas_to_coords(x, y))
                val_loss += loss.item() * x.size(0)

        val_loss /= val_size

        if save_model and epoch % 50 == 0 and epoch > 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                model,
                os.path.join("checkpoints", f"{save_model}_{epoch}.pth"),
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
            pred = model(x)
            loss = criterion(deltas_to_coords(x, pred), deltas_to_coords(x, y))
            test_loss += loss.item() * x.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}", flush=True)
    return test_loss


def train_model_from_dataset(
    k=100,
    epochs=100,
    ds_diff_in_seq=15,
    ds_window_total=420,
    ds_window_pred=120,
    ds_stride=15,
    training_batchsize=32,
    training_lr=1e-4,
    save_model=False,
):
    df = load_parquet("dataset", k=k)
    df = preprocess_data(df)

    dataset = SlidingWindowDatasetTemporal(
        df,
        max_diff_per_sequence_minutes=ds_diff_in_seq,
        window_size_minutes=ds_window_total,
        pred_size_minutes=ds_window_pred,
        stride=ds_stride,
    )
    print(f"Total dataset size: {len(dataset)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...", flush=True)
    model = TempTPI(pred_len=dataset[0][1].shape[0]).to(device)

    trained_model, train_loader, val_loader, test_loader = _train(
        model,
        dataset,
        device,
        batch_size=training_batchsize,
        lr=training_lr,
        epochs=epochs,
        save_model=save_model,
    )

    _evaluate(trained_model, test_loader, device)

    if save_model:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            trained_model,
            os.path.join(
                "checkpoints",
                f"{save_model}.pth",
            ),
        )
        # to load later:
        # model = torch.load(filepath)
        # model.eval()

    return trained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ds_diff_in_seq", type=int, default=15)
    parser.add_argument("--ds_window_total", type=int, default=420)
    parser.add_argument("--ds_window_pred", type=int, default=120)
    parser.add_argument("--ds_stride", type=int, default=15)
    parser.add_argument("--training_batchsize", type=int, default=32)
    parser.add_argument("--training_lr", type=float, default=1e-4)
    args = parser.parse_args()

    savename = False
    savename = f"temptpidec_{args.k}_{args.epochs}_{args.ds_diff_in_seq}_{args.ds_window_total}_{args.ds_window_pred}_{args.ds_stride}_{args.training_batchsize}_{args.training_lr}"

    argnames = [
        "k",
        "epochs",
        "DS: Max timediff in seq",
        "DS: Total window size",
        "DS: Prediction window size",
        "DS: Stride",
        "Training: Batch size",
        "Training: Learning rate",
    ]
    print(
        "Starting training for TempTPI with the following parameters:\n\t"
        + "\n\t".join(a + " - " + b for a, b in zip(argnames, savename.split("_")[1:]))
        + "\n"
    )

    model = train_model_from_dataset(
        model=args.model,
        k=args.k,
        epochs=args.epochs,
        ds_diff_in_seq=args.ds_diff_in_seq,
        ds_window_total=args.ds_window_total,
        ds_window_pred=args.ds_window_pred,
        ds_stride=args.ds_stride,
        training_batchsize=args.training_batchsize,
        training_lr=args.training_lr,
        save_model=savename,
    )
