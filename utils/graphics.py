# Plot training progress
import matplotlib.pyplot as plt
import csv


paths = [
    "./training_log_tfdecay.csv",
    "./training_log_sampled_tf.csv",
]

def plot_training_progress(csv_path):

    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(
            f, fieldnames=["timestamp", "step", "epoch", "loss", "lr"]
        )
        rows = [row for row in reader]
        rows.pop(0)  # Remove header row if present
    plt.plot(
        [int(row["step"]) for row in rows],
        [float(row["loss"]) for row in rows],
        marker="o",
        linewidth=2,
        markersize=6,
        label=csv_path,
    )


plt.figure(figsize=(10, 6))

for csv_path in paths:
    plot_training_progress(csv_path)

plt.xlabel("Step", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.title("Training Loss Over Time", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()
