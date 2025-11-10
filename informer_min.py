
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Utilities
# =========================

def _with_positional_encoding(x: torch.Tensor, max_len: int) -> torch.Tensor:
    # Add classic sinusoidal positional encoding. x: [B, T, D]
    B, T, D = x.shape
    pe = torch.zeros(max_len, D, device=x.device)
    position = torch.arange(0, max_len, device=x.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, D, 2, device=x.device) * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return x + pe[:T].unsqueeze(0)


# =========================
# ProbSparse Attention (toy but faithful in spirit)
# =========================

class ProbSparseAttention(nn.Module):
    # Simplified ProbSparse attention:
    # - Select top-u queries (by query norm)
    # - Compute full attention for selected queries
    # - Others use a lightweight context vector (avg of V)
    # Complexity ~ O(u*N) with u < N.
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask: Optional[torch.Tensor] = None, top_u: Optional[int] = None):
        # Q, K, V: [B, H, Tq, Dh], [B, H, Tk, Dh], [B, H, Tk, Dh]
        # attn_mask: [B, 1, Tq, Tk] or [B, H, Tq, Tk] (1 for mask, 0 for keep) - optional
        # top_u: number of queries to select. If None, use ceil(log(Tk)) * 16 heuristic.
        B, H, Tq, Dh = Q.shape
        Tk = K.shape[2]

        if top_u is None:
            top_u = max(1, min(Tq, int(math.ceil(math.log(Tk + 1) * 16))))

        # query importance score
        q_score = Q.pow(2).sum(dim=-1)  # [B, H, Tq]
        top_idx = torch.topk(q_score, k=top_u, dim=-1).indices  # [B, H, top_u]

        # context: mean of V per head
        context = V.mean(dim=2, keepdim=True)  # [B, H, 1, Dh]
        context = context.expand(-1, -1, Tq, -1).contiguous()  # [B, H, Tq, Dh]
        out = context.clone()

        scale = 1.0 / math.sqrt(Dh)
        for b in range(B):
            for h in range(H):
                sel = top_idx[b, h]  # [top_u]
                q_sel = Q[b:b+1, h:h+1, sel, :]  # [1,1,u,Dh]
                scores = torch.matmul(q_sel, K[b:b+1, h:h+1].transpose(-2, -1)) * scale  # [1,1,u,Tk]

                if attn_mask is not None:
                    m = attn_mask[b:b+1, :, sel, :]
                    scores = scores.masked_fill(m.bool(), float('-inf'))

                attn = torch.softmax(scores, dim=-1)
                attn = self.dropout(attn)
                o_sel = torch.matmul(attn, V[b:b+1, h:h+1])  # [1,1,u,Dh]
                out[b, h, sel, :] = o_sel[0, 0]

        return out  # [B, H, Tq, Dh]


class MultiHeadProbSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, top_u: Optional[int] = None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_u = top_u

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn = ProbSparseAttention(dropout=dropout)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, attn_mask: Optional[torch.Tensor] = None):
        # x_q: [B,Tq,D], x_kv: [B,Tk,D]
        B, Tq, D = x_q.shape
        Tk = x_kv.shape[1]

        Q = self.q_proj(x_q).view(B, Tq, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.k_proj(x_kv).view(B, Tk, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.v_proj(x_kv).view(B, Tk, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B,1,Tq,Tk]
        out = self.attn(Q, K, V, attn_mask=attn_mask, top_u=self.top_u)  # [B,H,Tq,Dh]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Tq, D)
        out = self.o_proj(out)
        return self.dropout(out)


# =========================
# Encoder with Distilling
# =========================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=512, dropout=0.1, top_u=None):
        super().__init__()
        self.self_attn = MultiHeadProbSparseAttention(d_model, n_heads, dropout=dropout, top_u=top_u)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        sa = self.self_attn(x, x, attn_mask=attn_mask)
        x = self.norm1(x + sa)
        ff = self.ff(x)
        x = self.norm2(x + ff)
        return x


class ConvDistill(nn.Module):
    # Informer-style distilling: Conv1d + GELU + stride-2 downsampling.
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, stride=2)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x: [B,T,D] -> [B,D,T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x  # [B, T//2, D]


class InformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff=512, dropout=0.1, top_u=None, distill=True):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff=d_ff, dropout=dropout, top_u=top_u)
            for _ in range(n_layers)
        ])
        self.distill = distill
        if distill:
            self.distills = nn.ModuleList([ConvDistill(d_model) for _ in range(n_layers - 1)])
        else:
            self.distills = None

    def forward(self, x, attn_mask=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, attn_mask=attn_mask)
            if self.distill and i < len(self.layers) - 1:
                x = self.distills[i](x)
        return x  # [B,T',D]


# =========================
# Decoder
# =========================

class MultiHeadFullAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, attn_mask: Optional[torch.Tensor] = None):
        B, Tq, D = x_q.shape
        Tk = x_k.shape[1]
        Q = self.q_proj(x_q).view(B, Tq, self.n_heads, self.d_head).permute(0,2,1,3)
        K = self.k_proj(x_k).view(B, Tk, self.n_heads, self.d_head).permute(0,2,1,3)
        V = self.v_proj(x_v).view(B, Tk, self.n_heads, self.d_head).permute(0,2,1,3)
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2,-1)) * scale  # [B,H,Tq,Tk]
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores.masked_fill(attn_mask.bool(), float('-inf'))
        A = torch.softmax(scores, dim=-1)
        A = self.dropout(A)
        out = torch.matmul(A, V)  # [B,H,Tq,Dh]
        out = out.permute(0,2,1,3).contiguous().view(B, Tq, D)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadFullAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadFullAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, mem, self_mask=None, cross_mask=None):
        sa = self.self_attn(x, x, x, attn_mask=self_mask)
        x = self.norm1(x + sa)
        ca = self.cross_attn(x, mem, mem, attn_mask=cross_mask)
        x = self.norm2(x + ca)
        ff = self.ff(x)
        x = self.norm3(x + ff)
        return x


class InformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, mem, self_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, mem, self_mask=self_mask, cross_mask=cross_mask)
        return x


# =========================
# Informer Model (end-to-end)
# =========================

class Informer(nn.Module):
    def __init__(self, c_in, c_out, seq_len, label_len, pred_len,
                 d_model=256, n_heads=4, e_layers=2, d_layers=1, d_ff=512,
                 dropout=0.1, top_u=None, distill=True, max_len=4096):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        # Embeddings
        self.enc_in = nn.Linear(c_in, d_model)
        self.dec_in = nn.Linear(c_in, d_model)

        self.encoder = InformerEncoder(d_model, n_heads, e_layers, d_ff=d_ff, dropout=dropout, top_u=top_u, distill=distill)
        self.decoder = InformerDecoder(d_model, n_heads, d_layers, d_ff=d_ff, dropout=dropout)
        self.proj_out = nn.Linear(d_model, c_out)
        self.max_len = max_len

    def _causal_mask(self, B, T):
        # Upper-triangular mask (1 above diagonal = masked)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        return mask.unsqueeze(0).expand(B, -1, -1)  # [B,T,T]

    def forward(self, enc_x, dec_x):
        # enc_x: [B, seq_len, c_in] (history)
        # dec_x: [B, label_len+pred_len, c_in] (teacher forcing tokens + known future covariates)
        B = enc_x.size(0)

        # Encode
        e = self.enc_in(enc_x)
        e = _with_positional_encoding(e, max_len=self.max_len)
        mem = self.encoder(e)  # [B, Tenc', D]

        # Decode
        d = self.dec_in(dec_x)
        d = _with_positional_encoding(d, max_len=self.max_len)
        self_mask = self._causal_mask(B, d.size(1)).to(d.device)
        out = self.decoder(d, mem, self_mask=self_mask, cross_mask=None)  # [B, Tdec, D]

        # Project only the last pred_len steps
        out = self.proj_out(out[:, -self.pred_len:, :])  # [B, pred_len, c_out]
        return out


# =========================
# Tiny synthetic demo
# =========================

def make_sine_data(B=32, seq_len=96, label_len=24, pred_len=24, c_in=1, device='cpu'):
    # Toy dataset with seasonality + trend + noise.
    # Returns (enc_x, dec_x, y_true)
    T_total = seq_len + pred_len
    t = torch.arange(T_total, device=device).float()[None, :, None]  # [1,T,1]
    freq = 2 * math.pi / 24.0
    base = torch.sin(freq * t) + 0.5 * torch.sin(freq * 2 * t + 0.3)
    trend = 0.002 * t
    noise = 0.05 * torch.randn(1, T_total, c_in, device=device)
    series = base + trend + noise  # [1, T, 1]
    series = series.expand(B, -1, c_in).contiguous()

    enc_x = series[:, :seq_len, :]
    y_true = series[:, -pred_len:, :]

    # decoder input: last label_len observed values in the first segment; zeros elsewhere
    dec_x = torch.zeros(B, label_len + pred_len, c_in, device=device)
    ctx = enc_x[:, -label_len:, :]
    dec_x[:, :label_len, :] = ctx

    return enc_x, dec_x, y_true


def quick_train_step(device='cpu'):
    torch.manual_seed(42)
    B, seq_len, label_len, pred_len, c_in = 16, 96, 24, 24, 1

    model = Informer(
        c_in=c_in, c_out=c_in, seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        d_model=128, n_heads=4, e_layers=2, d_layers=1, d_ff=256, dropout=0.1,
        top_u=None, distill=True
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.L1Loss()

    model.train()
    enc_x, dec_x, y_true = make_sine_data(B=B, seq_len=seq_len, label_len=label_len, pred_len=pred_len, c_in=c_in, device=device)

    # run a few steps to verify training signal
    losses = []
    for step in range(50):
        opt.zero_grad()
        y_hat = model(enc_x, dec_x)  # [B, pred_len, c_in]
        loss = loss_fn(y_hat, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    return losses, y_hat.detach().cpu(), y_true.detach().cpu()


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     losses, y_hat, y_true = quick_train_step(device=device)
#     print("Final loss:", losses[-1])
