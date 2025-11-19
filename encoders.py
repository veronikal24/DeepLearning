import torch
import torch.nn as nn


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
