# Plot training progress
import matplotlib.pyplot as plt
import csv


paths = [
    # "training_log100k.csv",
    # "training_log_tfdecay.csv",
    # "training_log_sampled_tf.csv",
    # "training_log_sampled_tf_small_model.csv",
    "training_log_sampled_tf_mid_model.csv",
]


def plot_training_progress(csv_path, smooth_window=1):
    """
    Plot training progress from CSV file.
    
    Args:
        csv_path: Path to training log CSV
        smooth_window: Number of steps to average over (1 = no smoothing)
    """
    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(
            f, fieldnames=["timestamp", "step", "epoch", "loss", "lr", "tf_ratio", "val_loss"]
        )
        rows = [row for row in reader]
        rows.pop(0)  # Remove header row if present
    
    total_rows = len(rows)
    rows = rows[smooth_window-1::smooth_window]
    for r in rows:
        print(f"({int(r['step'])//1000}, {r['val_loss']:.4}) ", end="")
        
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    
    plt.plot(
        steps,
        losses,
        marker="o",
        linewidth=2,
        markersize=6,
        label=csv_path.removesuffix(".csv"),
    )


plt.figure(figsize=(10, 6))

# Adjust smooth_window to smooth over n steps (e.g., 5, 10, 50)
smooth_window = 1  # Set to > 1 to average over that many steps

for csv_path in paths[1:]:
    plot_training_progress(csv_path, smooth_window=smooth_window)
plot_training_progress(paths[0], smooth_window=6)

plt.xlabel("Step", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.title("Training Loss Over Time", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()
