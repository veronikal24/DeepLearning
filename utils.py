import torch
import numpy as np
import torch.nn as nn
from global_land_mask import globe  # assuming `global-land-mask` is installed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load contents from globe package once
_land_mask_np = np.logical_not(globe._mask)  # converts ocean-mask to land-mask
_land_mask = torch.from_numpy(_land_mask_np)  # shape [21600, 43200], dtype=bool
_land_mask = _land_mask.to(device)  # stays on GPU permanently

_lat_vals = torch.from_numpy(globe._lat).float().to(device)
_lon_vals = torch.from_numpy(globe._lon).float().to(device)

lat0, dlat = _lat_vals[0], _lat_vals[1] - _lat_vals[0]
lon0, dlon = _lon_vals[0], _lon_vals[1] - _lon_vals[0]


def deltas_to_coords(
    input_tensor: torch.Tensor, pred_deltas: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the absolute coords (in lat/lon) for a given pred_deltas tensor, based on the coords in the given input_tensor.
    """
    # get the last known absolute coordinates from input
    last_abs = input_tensor[:, -1, 2:4]  # shape [B, 2]

    # compute cumulative sum of deltas across time dimension
    abs_coords = last_abs.unsqueeze(1) + torch.cumsum(pred_deltas, dim=1)

    return abs_coords


def latlon_to_index(lat, lon):
    # clamp
    lat_i = torch.clamp((lat - lat0) / dlat, 0, _lat_vals.numel() - 1)
    lon_i = torch.clamp((lon - lon0) / dlon, 0, _lon_vals.numel() - 1)

    return lat_i.long(), lon_i.long()


def land_mask_from_coords(pred_coords):
    # pred_coords: [B, T, 2], lat first
    lat = pred_coords[..., 0]
    lon = pred_coords[..., 1]

    lat_i, lon_i = latlon_to_index(lat, lon)
    return _land_mask[lat_i, lon_i]  # returns [B, T] boolean mask


class PenalizedCoordLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, penalty_value=100):
        super().__init__()
        self.base_loss = base_loss
        self.penalty_value = penalty_value

    def forward(self, pred_coords, target_coords):
        base = self.base_loss(pred_coords, target_coords)

        land_mask = land_mask_from_coords(pred_coords)  # [B, T], bool, GPU

        # Broadcast to [B, T, 2]
        penalty = land_mask.unsqueeze(-1).float() * self.penalty_value
        return base + penalty.mean()


class WeightedStepMSELoss(nn.Module):
    """
    MSE loss with increasing weights for later steps in a sequence.

    mode: 'linear' or 'exponential' weighting. Default: Equal weighting (None).
    start_weight: Lowest assigned weight
    end_weight: Highest assigned weight
    """

    def __init__(self, mode="", start_weight=1.0, end_weight=3.0):
        super().__init__()
        self.mode = mode
        self.start_weight = start_weight
        self.end_weight = end_weight

    def forward(self, y_pred, y_true):
        # y_pred, y_true: (batch_size, seq_len, features)
        batch_size, seq_len, n_features = y_pred.shape

        # Create weights along the sequence dimension
        if self.mode.startswith("lin"):
            weights = torch.linspace(
                self.start_weight, self.end_weight, steps=seq_len, device=y_pred.device
            )
        elif self.mode.startswith("exp"):
            weights = torch.exp(
                torch.linspace(
                    self.start_weight,
                    self.end_weight,
                    steps=seq_len,
                    device=y_pred.device,
                )
            )
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
