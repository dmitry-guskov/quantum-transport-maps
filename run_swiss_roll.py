"""
Swiss Roll (uniform -> spiral) experiment.

Usage:
    python run_swiss_roll.py                  # train from scratch
    python run_swiss_roll.py --load VERSION   # load saved thetas, skip training
"""
import argparse
import os

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from config import CircuitConfig, TrainConfig, InferConfig
from circuit import build_circuit
from distributions import uniform_to_spiral
from training import train
from inference import infer, evaluate_ot
from utils import plot_pairs


class XYDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default=None, help="Version suffix to load (e.g. '-v3_worksgreat')")
    parser.add_argument("--save-version", type=str, default="-v1", help="Version suffix for saving")
    parser.add_argument("--data-dir", type=str, default="uniform_to_spiral", help="Data directory")
    args = parser.parse_args()

    # --- Configuration ---
    circuit_cfg = CircuitConfig(
        circuit_type="v3",
        n_qubits=2,
        n_copies=2,
        n_ansatzlayers_body=3,
        n_ansatzlayers_tail=3,
        n_reuploads=15,
        n_lines=2,
        entangler_type="l",
        measurement_qubits="all",
        data_dim=2,
    )
    train_cfg = TrainConfig(n_epochs=400, batch_size=128, lr=0.01, alpha=0.8, beta=0.2, r_rate=2000.0, seed=42)
    infer_cfg = InferConfig(n_epochs=200, lr=0.05, batch_size=256)

    save_dir = args.data_dir
    os.makedirs(f"{save_dir}/optimal_thetas", exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    os.makedirs(f"{save_dir}/dataset", exist_ok=True)

    # --- Data ---
    torch.manual_seed(train_cfg.seed)

    data_path_x = f"{save_dir}/dataset/x_data.pt"
    data_path_y = f"{save_dir}/dataset/y_data.pt"

    if os.path.exists(data_path_x) and os.path.exists(data_path_y):
        print("Loading existing dataset...")
        x_data = torch.load(data_path_x, weights_only=True)
        y_data = torch.load(data_path_y, weights_only=True)
    else:
        print("Generating new dataset...")
        x_data, y_data = uniform_to_spiral(num_samples=train_cfg.train_size + train_cfg.test_size)
        torch.save(x_data, data_path_x)
        torch.save(y_data, data_path_y)

    x_train, y_train = x_data[: train_cfg.train_size], y_data[: train_cfg.train_size]
    x_test, y_test = x_data[train_cfg.train_size : train_cfg.train_size + train_cfg.test_size], y_data[train_cfg.train_size : train_cfg.train_size + train_cfg.test_size]

    dataloader_train = DataLoader(XYDataset(x_train, y_train), batch_size=train_cfg.batch_size, shuffle=True)
    print(f"Train: {len(x_train)}, Test: {len(x_test)}, data_dim: {x_train.shape[1]}")

    # --- Build circuit ---
    circuit, n_thetas, info = build_circuit(circuit_cfg)
    print(f"Circuit: {info}")
    print(f"Parameters: {n_thetas}")

    # --- Train or load ---
    if args.load is not None:
        theta_path = f"{save_dir}/optimal_thetas/optimal_thetas_{info}{args.load}.pt"
        print(f"Loading thetas from {theta_path}")
        thetas = torch.load(theta_path, weights_only=True)
        assert len(thetas) == n_thetas, f"Shape mismatch: {len(thetas)} vs {n_thetas}"
        loss_history = None
    else:
        print("Training...")
        thetas = torch.rand(n_thetas, requires_grad=True)
        thetas, loss_history = train(circuit, thetas, dataloader_train, train_cfg)

        theta_path = f"{save_dir}/optimal_thetas/optimal_thetas_{info}{args.save_version}.pt"
        torch.save(thetas, theta_path)
        print(f"Saved thetas to {theta_path}")

    # --- Inference ---
    print("Running inference...")
    y_star, eval_loss, y_star_history = infer(circuit, thetas, x_test, infer_cfg)

    ot_dist = evaluate_ot(y_star, y_test)
    print(f"OT distance: {ot_dist:.6f}")

    # --- Plots ---
    n_plots = 3 if loss_history is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    ax_idx = 0
    if loss_history is not None:
        axes[ax_idx].plot(loss_history)
        axes[ax_idx].set_xlabel("Epoch")
        axes[ax_idx].set_ylabel("Loss")
        axes[ax_idx].set_title("Training Loss")
        ax_idx += 1

    plot_pairs(x_test, y_test, ax=axes[ax_idx], colors=["blue", "red"])
    axes[ax_idx].set_title("Source (blue) / Target (red)")

    ax_idx += 1
    plot_pairs(x_test, y_star, ax=axes[ax_idx], colors=["blue", "green"])
    axes[ax_idx].set_title(f"Source (blue) / Predicted (green)\nOT: {ot_dist:.4f}")

    plt.tight_layout()
    plot_path = f"{save_dir}/plots/swiss_roll_{info}{args.save_version if args.load is None else args.load}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
