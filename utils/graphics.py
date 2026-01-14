# Plot training progress
import matplotlib.pyplot as plt
import csv


csv_path = './training_log.csv'

with open(csv_path, mode="r") as f:
    reader = csv.DictReader(f, fieldnames=["timestamp", "step", "epoch", "loss", "lr"])
    rows = [row for row in reader]
    rows.pop(0)  # Remove header row if present

nthrows = rows[::50]

plt.figure(figsize=(10, 6))
plt.plot([int(row["step"]) for row in nthrows], [float(row["loss"]) for row in nthrows], marker='o', linewidth=2, markersize=6)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Training Summary:")
print(f"Initial Loss: {float(rows[0]['loss']):.4f}")
print(f"Final Loss: {float(rows[-1]['loss']):.4f}")
print(f"Steps: {int(rows[-1]['step']):,}")
print(f"Best Loss: {min(float(row['loss']) for row in rows):.4f}")
print(f"Improvement: {((float(rows[0]['loss']) - float(rows[-1]['loss'])) / float(rows[0]['loss']) * 100):.1f}%")