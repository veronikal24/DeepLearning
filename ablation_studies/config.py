"""
Ablation Study Configuration for TPTrans

Parameters to test:
- window_size_minutes: [360, 480]      (6h, 8h)
- pred_size_minutes: [60, 120, 180, 240, 300]   (1h, 2h, 3h, 4h, 5h)
- stride: [15, 30]         (15min, 30min)
- k: [100, 500, 1000]   (number of MMSIs)

Training configuration:
- epochs: 500 with early stopping (patience=20)
- learning_rate: 1e-4
- batch_size: 32
- model: d_model=256, nhead=8, num_encoder_layers=8
"""

# Parameter ranges (use loops to generate all combinations)
# window_size_minutes = [240, 360, 480]  # 4h, 6h, 8h
# pred_size_minutes = [60, 120, 180, 240, 300]  # 1h, 2h, 3h, 4h, 5h
# stride = [15, 30]  # 15min, 30min
# k = [100, 500, 1000]  # MMSIs

# Comment Kevin: I think we shouldn't create ALL combinations, instead do different blocks
# f.ex. block 1 (observe prediction quality based on in- and output timeframe):
window_size_minutes = [240, 360, 480]  # 4h, 6h, 8h
pred_size_minutes = [60, 120, 180, 240, 300]  # 1h, 2h, 3h, 4h, 5h
stride = [30]
k = [1000]  # MMSIs
# and block 2 (observe prediction quality based on larger/smaller datasets):
# window_size_minutes = [480]  # 8h
# pred_size_minutes = [300]  # 5h
# stride = [15, 30]  # 15min, 30min
# k = [100, 500, 1000]  # MMSIs

# Training config
epochs = 300
early_stopping_patience = 20

# Output metrics
output_columns = [
    "window_size_minutes",
    "pred_size_minutes",
    "stride",
    "k",
    "epochs_trained",
    "early_stopped",
    "train_loss",
    "val_loss",
    "test_loss",
    "training_time_s",
    "timestamp",
]
