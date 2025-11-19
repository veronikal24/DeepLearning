"""
Ablation Study Runner for TPTrans

Executes all parameter combinations from config.py and logs results to CSV.
"""

import csv
import os
import sys
import time
import argparse
from datetime import datetime
from itertools import product

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # Add parent dir to path

from training import train_model_from_dataset
from tptrans import TPTrans
from tpinform import TPInform
from temptpi import TempTPT, TempTPI
from dataloader import SlidingWindowDataset, SlidingWindowDatasetTemporal
from config import (
    window_size_minutes,
    pred_size_minutes,
    stride,
    k,
    epochs,
    early_stopping_patience,
    output_columns,
)


def main(model="TPTrans"):
    """Run all ablation experiments."""
    if model == "TPTrans":
        M = TPTrans
    elif model == "TPInform":
        M = TPInform
    elif model == "TempTPT":
        M = TempTPT
    elif model == "TempTPI":
        M = TempTPI
    else:
        print("Invalid model provided :(")
        raise Exception
    if model in ["TPTrans", "TPInform"]:
        D = SlidingWindowDataset
    else:
        D = SlidingWindowDatasetTemporal

    results_file = "results.csv"

    # Initialize CSV
    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=output_columns).writeheader()

    combinations = list(product(window_size_minutes, pred_size_minutes, stride, k))
    print(f"\nRunning {len(combinations)} experiments for {M.__name__}...\n")

    for i, (w, p, s, k_val) in enumerate(combinations, 1):
        print(
            f"[{i}/{len(combinations)}] window={w + p}, pred={p}, stride={s}, k={k_val}"
        )
        start_time = time.time()

        try:
            trained_model, test_loss, history = train_model_from_dataset(
                Model=M,  # choose TPTrans, TPInform, TempTPT or TempTPI
                Dataset=D,  # choose SlidingWindowDataset for TPTrans and TPInform, or SlidingWindowDatasetTemporal for TempTP*
                k=k_val,
                epochs=epochs,
                ds_diff_in_seq=15,
                ds_window_total=w + p,
                ds_window_pred=p,
                ds_stride=s,
                training_batchsize=32,
                training_lr=1e-4,
                save_model=False,
                early_stopping_patience=early_stopping_patience,
                dataset_path="../dataset",
            )
            training_time = time.time() - start_time
            epochs_trained = history["epochs_trained"]
            early_stopped = (
                1 if (early_stopping_patience and epochs_trained < epochs) else 0
            )

            result = {
                "window_size_minutes": w + p,
                "pred_size_minutes": p,
                "stride": s,
                "k": k_val,
                "epochs_trained": epochs_trained,
                "early_stopped": early_stopped,
                "train_loss": f"{history['train_loss']:.6f}",
                "val_loss": f"{history['val_loss']:.6f}",
                "test_loss": f"{test_loss:.6f}",
                "training_time_s": f"{training_time:.2f}",
                "timestamp": datetime.now().isoformat(),
            }
            print(
                f"train={result['train_loss']}, val={result['val_loss']}, test={result['test_loss']}\n"
            )

        except Exception as e:
            training_time = time.time() - start_time
            result = {
                "window_size_minutes": w + p,
                "pred_size_minutes": p,
                "stride": s,
                "k": k_val,
                "epochs_trained": 0,
                "early_stopped": 0,
                "train_loss": "ERROR",
                "val_loss": "ERROR",
                "test_loss": "ERROR",
                "training_time_s": f"{training_time:.2f}",
                "timestamp": datetime.now().isoformat(),
            }
            print(f"{str(e)}\n")

        with open(results_file, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=output_columns).writerow(result)

    print(f"Done, results in {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument("--model", type=str, default="TPTrans")
    args = parser.parse_args()
    main(model=args.model)
