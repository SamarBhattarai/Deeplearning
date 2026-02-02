import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Data
n_points = 100000

# Uniform: Random numbers between 0 and 1
data_uniform = np.random.rand(n_points)

# Gaussian: Random numbers centered at 0
data_normal = np.random.randn(n_points)

# 2. Plot
plt.figure(figsize=(12, 5))

# --- Plot 1: Uniform Distribution ---
plt.subplot(1, 2, 1)
# NOTE: We use plt.hist, not plt.scatter, to see the "Shape"
plt.hist(data_uniform, bins=50, color='blue', edgecolor='black')
plt.title("np.random.rand()\n(Uniform Distribution)")
plt.xlabel("Value")
plt.ylabel("Frequency (Count)")
# Text is placed at y=2000, which fits the FREQUENCY scale
plt.text(0.5, 2100, "FLAT SHAPE", ha='center', fontsize=12, color='red', weight='bold')

# --- Plot 2: Gaussian Distribution ---
plt.subplot(1, 2, 2)
plt.hist(data_normal, bins=50, color='red', edgecolor='black')
plt.title("np.random.randn()\n(Normal/Gaussian Distribution)")
plt.xlabel("Value")
plt.ylabel("Frequency (Count)")
# Text is placed at y=2000, which fits the FREQUENCY scale
plt.text(0, 2100, "BELL CURVE", ha='center', fontsize=12, color='blue', weight='bold')

plt.tight_layout()
plt.show()