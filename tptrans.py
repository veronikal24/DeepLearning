import torch
import torch.nn as nn
import argparse
from dataloader import SlidingWindowDataset
from encoders import PositionalEncoding
from training import train_model_from_dataset


class TPTrans(nn.Module):
    def __init__(
        self,
        input_dim=6,
        d_model=256,
        nhead=8,
        num_encoder_layers=8,
        pred_len=6,
        conv_kernel=3,
        conv_padding=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len

        # project 6 inputs to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

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

        # project input to d_model
        x = self.input_proj(src)  # (batch, src_len, d_model)

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

        # combine context + positional encoding
        context_expanded = context.unsqueeze(0).expand(pred_len, -1, -1)
        dec_in = context_expanded + tgt_pos

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
    savename = f"tprans_{args.k}_{args.epochs}_{args.ds_diff_in_seq}_{args.ds_window_total}_{args.ds_window_pred}_{args.ds_stride}_{args.training_batchsize}_{args.training_lr}"

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
        "Starting training for TPTrans with the following parameters:\n\t"
        + "\n\t".join(a + " - " + b for a, b in zip(argnames, savename.split("_")[1:]))
        + "\n"
    )

    model = train_model_from_dataset(
        Model=TPTrans,
        Dataset=SlidingWindowDataset,
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
