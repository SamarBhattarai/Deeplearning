import numpy as np
import matplotlib.pyplot as plt

def compute_ewa(data, beta, bias_correction=False):
    """
    Computes EWA with optional Bias Correction.
    """
    v = 0
    ewa_values = []
    
    # We  are using 'enumerate' to get the time step 't' (starting at 1)
    for t, theta in enumerate(data, start=1):
        
        # Standard Update
        v = (beta * v) + (1 - beta) * theta
        
        if bias_correction:
            # Apply Correction
            correction_factor = 1 - (beta ** t)
            v_corrected = v / correction_factor
            ewa_values.append(v_corrected)
        else:
            ewa_values.append(v)
            
    return np.array(ewa_values)

# Data

# Creating 100 days
days = np.arange(1, 101)

np.random.seed(1)

original_wave = 20 + 5 * np.sin(days/10) # Smooth sine wave
noise = np.random.randn(100) * 2 # Inorder to add noise
data_noisy = original_wave + noise
# print(data_noisy)
# print(data_noisy.shape)

### CALCULATING EWA WITH DIFFERENT BETA VALUES ###

# Beta = 0.5 (Low memory: Averages ~2 days)
# Very jagged, follows noise closely.
ewa_low = compute_ewa(data_noisy, beta=0.5)

# Beta = 0.9 (Standard memory: Averages ~10 days)
# Good balance of smoothness and accuracy.
ewa_std = compute_ewa(data_noisy, beta=0.9)

ewa_std_with_bias = compute_ewa(data_noisy, beta = 0.9, bias_correction = True)

# Beta = 0.98 (High memory: Averages ~50 days)
# Very smooth, but "lags" behind the real trend.
ewa_high = compute_ewa(data_noisy, beta=0.98)

# --- 3. VISUALIZATION ---
plt.figure(figsize=(12, 6))

# Plot Raw Noisy Data
plt.scatter(days, data_noisy, alpha=0.3, color='gray', label='Raw Noisy Data')

# Plot EWAs
plt.plot(days, ewa_low, color='yellow', linestyle='--', linewidth=2, label='Beta = 0.5 (Noisy)')
plt.plot(days, ewa_std, color='red', linewidth=3, label='Beta = 0.9 (Balanced)')
plt.plot(days, ewa_std_with_bias, color='green', linewidth=3, label='Beta = 0.9 (Balanced_with_bias)')
plt.plot(days, ewa_high, color='blue', linewidth=2, label='Beta = 0.98 (Laggy)')

plt.title("Exponentially Weighted Averages (EWA) Intuition")
plt.xlabel("Days")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

