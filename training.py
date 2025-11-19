import os
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from dataloader import load_parquet, preprocess_data
from utils import deltas_to_coords


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
    early_stopping_patience=None,
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

    best_val_loss = float("inf")
    no_improve = 0
    epochs_trained = 0
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
                os.path.join("checkpoints", f"{save_model}_{epoch}.pth"),
            )

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}",
            flush=True,
        )
        epochs_trained = epoch + 1
        # Early stopping (if enabled) (always do at least 100 epochs)
        if early_stopping_patience is not None and epochs_trained > 100:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                best_state = model.state_dict().copy()
            else:
                no_improve += 1

            if no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}", flush=True)
                model.load_state_dict(best_state)
                break

    return (
        model,
        train_loader,
        val_loader,
        test_loader,
        {
            "epochs_trained": epochs_trained,
            "train_loss": avg_loss,
            "val_loss": val_loss,
        },
    )


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
    Model,
    Dataset,
    k=100,
    epochs=100,
    ds_diff_in_seq=15,
    ds_window_total=420,
    ds_window_pred=120,
    ds_stride=15,
    training_batchsize=32,
    training_lr=1e-4,
    save_model=False,
    dataset_path="dataset",
    early_stopping_patience=None,
):
    df = load_parquet(dataset_path, k=k)
    df = preprocess_data(df)

    dataset = Dataset(
        df,
        max_diff_per_sequence_minutes=ds_diff_in_seq,
        window_size_minutes=ds_window_total,
        pred_size_minutes=ds_window_pred,
        stride=ds_stride,
    )
    print(f"Total dataset size: {len(dataset)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...", flush=True)
    model = Model(pred_len=dataset[0][1].shape[0]).to(device)

    trained_model, train_loader, val_loader, test_loader, history = _train(
        model,
        dataset,
        device,
        batch_size=training_batchsize,
        lr=training_lr,
        epochs=epochs,
        save_model=save_model,
        early_stopping_patience=early_stopping_patience,
    )

    test_loss = _evaluate(trained_model, test_loader, device)

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

    return trained_model, test_loss, history
