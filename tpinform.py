import os
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from dataloader import load_parquet, preprocess_data, SlidingWindowDataset
from informer_original_paper import EncoderLayer, DecoderLayer, Encoder, Decoder, AttentionLayer, ProbAttention, FullAttention, ConvLayer
import argparse
from utils import deltas_to_coords, PenalizedCoordLoss


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

    def forward(self, x, tgt_start = 0):
        seq_len = x.size(0)
        return x + self.pe[:seq_len].unsqueeze(1).to(x.device).to(x.dtype)


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
        label_len=5, 
             conv_kernel=3,
        conv_padding=1,    
    ):
        super().__init__()
        self.d_model = d_model
        
        
        self.input_proj = nn.Linear(input_dim, d_model)

        # positional encoding as first block
        self.pos_encoder = PositionalEncoding(d_model)

        # convolution for short-term feature extraction
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel, padding=conv_padding
        )

        self.label_len = label_len
        #imported from informer_original_paper.py
        d_ff = 512
        activation = "gelu"
        factor = 5
        Attn = ProbAttention
        dropout = 0.1
        enc_layers = [
            EncoderLayer(
                AttentionLayer(
                    Attn(False, factor, attention_dropout=dropout, output_attention=False),
                    d_model,
                    nhead,
                    mix=False,
                ),
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_encoder_layers)
        ]

        # optional distilling convs between encoder layers (Informer trick)
        conv_layers = [
            ConvLayer(d_model) for _ in range(num_encoder_layers - 1)
        ]

        self.encoder = Encoder(
            attn_layers=enc_layers,
            conv_layers=conv_layers,
            norm_layer=nn.LayerNorm(d_model),
        )

        #---------------------------------------------
        self.decoder_input_proj = nn.Linear(2, d_model)
        #imported from informer_original_paper.py
        dec_layers = [
                    DecoderLayer(
                        self_attention=AttentionLayer(
                            Attn(True, factor, attention_dropout=dropout, output_attention=False),
                            d_model,
                            nhead,
                            mix=True,
                        ),
                        cross_attention=AttentionLayer(
                            FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                            d_model,
                            nhead,
                            mix=False,
                        ),
                        d_model=d_model,
                        d_ff=d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for _ in range(num_decoder_layers)
                ]
        self.pos_decoder = PositionalEncoding(d_model, max_len=2048)
        self.decoder = Decoder(
                    layers=dec_layers,
                    norm_layer=nn.LayerNorm(d_model),
                )
        
        #---------------------------------------------

        self.output_head = nn.Linear(d_model, 2)  # lat, lon
        self.pred_len = pred_len

    def forward(self, src, tgt_start):
        """
        src:       (B, src_len, input_dim)
        tgt_start: (B, 1, 2)   last known (lat, lon)
        """
        B, src_len, _ = src.shape
        pred_len = self.pred_len
        label_len = self.label_len

        # ---------- ENCODER ----------
        # 1) Project input to d_model
        x = self.input_proj(src)             # (B, L, d_model)

        # 2) Positional encoding (seq-first)
        x_seq = x.transpose(0, 1)            # (L, B, d_model)
        x_seq = self.pos_encoder(x_seq)      # (L, B, d_model)
        x = x_seq.transpose(0, 1)            # (B, L, d_model)

        # 3) Local convolution (Informer-style)
        x_conv = self.conv(x.transpose(1, 2))     # (B, d_model, L)
        x = x_conv.transpose(1, 2)                # (B, L, d_model)

        # 4) Informer encoder (batch-first)
        memory, _ = self.encoder(x, attn_mask=None)   # (B, L, d_model)

        # ---------- BUILD DECODER INPUT ----------
        dec_len = label_len + pred_len    # same as Informer

        # y in “lat/lon space”: (B, dec_len, 2)
        last_pos = tgt_start.squeeze(1)   # (B, 2)
        y = torch.zeros(B, dec_len, 2, device=src.device)

        # first part = context repeat
        y[:, :label_len, :] = last_pos.unsqueeze(1)

        # project to d_model
        tgt = self.decoder_input_proj(y)      # (B, dec_len, d_model)

        # add positional encoding (seq-first again)
        tgt_seq = tgt.transpose(0, 1)      # (dec_len, B, d_model)
        tgt_seq = self.pos_decoder(tgt_seq)   # (dec_len, B, d_model)
        tgt = tgt_seq.transpose(0, 1)         # (B, dec_len, d_model)

        # ---------- DECODER ----------
        dec_out = self.pos_decoder(tgt, memory)   # (B, dec_len, d_model)

        # output head → (lat, lon)
        dec_out = self.output_head(dec_out)       # (B, dec_len, 2)

        # keep only future part
        pred = dec_out[:, -pred_len:, :]          # (B, pred_len, 2)

        return pred






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
    criterion = PenalizedCoordLoss(nn.MSELoss())

    # criterion = PenalizedCoordLoss(nn.MSELoss())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            tgt_start = x[:, -1:, :2]  # grab lat/lon
            pred = model(x, tgt_start)
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
                tgt_start = x[:, -1:, :2]  # grab lat/lon
                pred = model(x, tgt_start)
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
            tgt_start = x[:, -1:, :2]  # grab lat/lon
            pred = model(x, tgt_start)
            loss = criterion(deltas_to_coords(x, pred), deltas_to_coords(x, y))
            test_loss += loss.item() * x.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}", flush=True)
    return test_loss


def train_model_from_dataset(
    k=100,
    epochs=100,
    ds_diff_in_seq=15,
    ds_window_total=120,
    ds_window_pred=30,
    ds_stride=15,
    training_batchsize=32,
    training_lr=1e-4,
    save_model=False,
):
    df = load_parquet("aisdk-2025-02-27", k=k)
    df = preprocess_data(df)

    dataset = SlidingWindowDataset(
        df,
        max_diff_per_sequence_minutes=ds_diff_in_seq,
        window_size_minutes=ds_window_total,
        pred_size_minutes=ds_window_pred,
        stride=ds_stride,
    )
    print(f"Total dataset size: {len(dataset)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...", flush=True)
    model = TPTrans(pred_len=dataset[0][1].shape[0]).to(device)

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
    savename = f"t_{args.k}_{args.epochs}_{args.ds_diff_in_seq}_{args.ds_window_total}_{args.ds_window_pred}_{args.ds_stride}_{args.training_batchsize}_{args.training_lr}"

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



