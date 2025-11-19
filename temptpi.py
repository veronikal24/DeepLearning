import numpy as np
import torch
import torch.nn as nn
import argparse
from dataloader import (
    SlidingWindowDatasetTemporal,
    SAMPLE_INTERVAL_MIN,
)
from informer_original_paper import (
    ProbAttention,
    EncoderLayer,
    AttentionLayer,
)
from encoders import PositionalEncoding, TemporalEncoding
from training import train_model_from_dataset


# This is the old class with just temporal encoding; For model with temporal + informer encoder, go to TempTPI
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
    savename = f"temptpi_{args.k}_{args.epochs}_{args.ds_diff_in_seq}_{args.ds_window_total}_{args.ds_window_pred}_{args.ds_stride}_{args.training_batchsize}_{args.training_lr}"

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

    model, loss, hist = train_model_from_dataset(
        Model=TempTPI,
        Dataset=SlidingWindowDatasetTemporal,
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
