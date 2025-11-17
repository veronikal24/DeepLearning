import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import random_split, DataLoader
from dataloader import load_parquet, preprocess_data, SlidingWindowDataset
from utils import deltas_to_coords, PenalizedCoordLoss


from informer_original_paper import (
    ProbAttention,
    FullAttention,
    AttentionLayer,
    EncoderLayer,
    DecoderLayer,
    Decoder,
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
        """
        Here we assume x is (seq_len, batch, d_model),
        because that matches typical Transformer-style PE.
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len].unsqueeze(1).to(x.device).to(x.dtype)


class TPTrans(nn.Module):
    def __init__(
        self,
        input_dim=6,
        d_model=256,
        nhead=8,
        num_encoder_layers=8,
        num_decoder_layers=2,
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

        # positional encoding for encoder and decoder
        self.pos_encoder = PositionalEncoding(d_model, max_len=2048)
        self.pos_decoder = PositionalEncoding(d_model, max_len=2048)

        # convolution for short-term feature extraction
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel, padding=conv_padding
        )

        # ===== Informer-style encoder =====
        d_ff = 512
        activation = "gelu"
        factor = 5
        dropout = 0.1
        Attn = ProbAttention

        self.encoder_layers = nn.ModuleList(
            [
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
        )

        # ===== Informer-style decoder =====
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

        self.decoder = Decoder(
            layers=dec_layers,
            norm_layer=nn.LayerNorm(d_model),
        )

        # final projection from model dimension to (delta_lat, delta_lon)
        self.out_proj = nn.Linear(d_model, 2)

    def forward(self, src, pred_len=None):
        """
        src: (batch, src_len, input_dim)
        returns: (batch, pred_len, 2) deltas
        """
        if pred_len is None:
            pred_len = self.pred_len

        batch, src_len, d_in = src.shape
        assert d_in == self.input_dim, f"Expected input_dim={self.input_dim}, got {d_in}"

        # ----- Encoder -----
        # project input to d_model
        x = self.input_proj(src)  # (batch, src_len, d_model)

        # positional encoding expects (seq_len, batch, d_model)
        x_pe = x.permute(1, 0, 2)  # (src_len, batch, d_model)
        x_pe = self.pos_encoder(x_pe)
        x = x_pe.permute(1, 0, 2)  # back to (batch, src_len, d_model)

        # convolution: conv1d expects (batch, channels, seq_len)
        x_conv = x.permute(0, 2, 1)  # (batch, d_model, src_len)
        x_conv = self.conv(x_conv)
        memory = x_conv.permute(0, 2, 1)  # (batch, src_len, d_model)

        # pass through encoder layers
        for layer in self.encoder_layers:
            # typical Informer EncoderLayer returns (enc_out, attn)
            memory, _ = layer(memory, attn_mask=None)  # memory: (batch, src_len, d_model)

        # ----- Decoder -----
        # create decoder input "query" sequence for pred_len steps
        # simplest: just zeros + positional encoding
        tgt = torch.zeros(
            batch, pred_len, self.d_model, device=src.device, dtype=memory.dtype
        )  # (batch, pred_len, d_model)

        # positional encoding on decoder inputs
        tgt_pe = tgt.permute(1, 0, 2)  # (pred_len, batch, d_model)
        tgt_pe = self.pos_decoder(tgt_pe)
        tgt = tgt_pe.permute(1, 0, 2)  # (batch, pred_len, d_model)

        # Informer-style decoder call:
        # decoder(x=tgt, cross=memory, x_mask=None, cross_mask=None)
        dec_out = self.decoder(
            tgt,
            memory,
            x_mask=None,
            cross_mask=None,
        )  # (batch, pred_len, d_model)

        # project to (delta_lat, delta_lon)
        preds = self.out_proj(dec_out)  # (batch, pred_len, 2)

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

        if save_model and epoch % 100 == 0 and epoch > 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                model,
                os.path.join("checkpoints", f"{save_model}_experiment_{epoch}.pth"),
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
    savename = f"tpinform_experiment_{args.k}_{args.epochs}_{args.ds_diff_in_seq}_{args.ds_window_total}_{args.ds_window_pred}_{args.ds_stride}_{args.training_batchsize}_{args.training_lr}"

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
