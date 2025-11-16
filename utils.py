import torch
import torch.nn as nn
from global_land_mask import globe  # assuming `global-land-mask` is installed


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


class PenalizedCoordLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, penalty_value=100):
        """
        Wraps a base loss (e.g., nn.MSELoss) and adds a fixed penalty
        for predictions that fall on land.

        base_loss: standard PyTorch loss function
        penalty_value: value to assign to coords on land
        """
        super().__init__()
        self.base_loss = base_loss
        self.penalty_value = penalty_value

    def forward(self, pred_coords: torch.Tensor, target_coords: torch.Tensor):
        """
        pred_coords: [B, T, 2] tensor with predicted [lat, lon]
        target_coords: [B, T, 2] tensor with target [lat, lon]
        """
        # Compute the base loss
        base_loss_val = self.base_loss(pred_coords, target_coords)

        # Handle land penalty safely
        B, T, _ = pred_coords.shape
        pred_np = pred_coords.detach().cpu().numpy().reshape(-1, 2)
        lat, lon = pred_np[:, 0], pred_np[:, 1]

        # Safe check for valid coordinates
        land_mask = torch.zeros(B * T, dtype=torch.bool)
        for i in range(B * T):
            try:
                if globe.is_land(lat[i], lon[i]):
                    land_mask[i] = True
            except Exception:
                # If invalid coordinate, ignore penalty
                pass

        land_mask = land_mask.reshape(B, T).to(pred_coords.device)

        # Compute a penalty tensor of same shape as coords
        penalty_tensor = torch.zeros_like(pred_coords)
        penalty_tensor[land_mask] = self.penalty_value

        # Add mean penalty to the base loss
        total_loss = base_loss_val + penalty_tensor.mean()

        return total_loss


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
