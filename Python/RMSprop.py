import numpy as np
import matplotlib.pyplot as plt

# --- SETTING UP THE FUNCTION (The "Canyon") ---
def get_gradients(x, y):
    """
    Function: f(x, y) = 0.1*x^2 + 2*y^2
    - X axis is flat (coefficient 0.1) -> Small gradients
    - Y axis is steep (coefficient 2)   -> Large gradients
    """
    dx = 0.2 * x # Small Gradient
    dy = 4 * y # Large Gradient

    return dx, dy

# Optimizers

def run_standard_sgd(start_x, start_y, lr, steps):
    path = []
    x, y = start_x, start_y

    for i in range(steps):
        path.append((x, y))
        dx, dy = get_gradients(x, y)

        # Standard Update
        x = x - (lr * dx)
        y = y - (lr * dy)

    return np.array(path)

def run_rmsprop(start_x, start_y, lr, beta, steps):
    path = []
    x, y = start_x, start_y

    # Initializing "S" (Squared Gradients Cache)
    s_x = 0
    s_y = 0
    epsilon = 1e-8 # Safety term to prevent division by zero

    for i in range(steps):
        path.append((x, y))
        dx, dy = get_gradients(x, y)

        # RMSprop algorithm
        s_x = (beta * s_x) + (1 - beta) * (dx ** 2)
        s_y = (beta * s_y) + (1 - beta) * (dy ** 2)

        # Updation
        # Note: high s_y will shrink the update for y!
        x = x - lr * dx / (np.sqrt(s_x) + epsilon)
        y = y - lr * dy / (np.sqrt(s_y) + epsilon)

    return np.array(path)

# Main 

start_pos = (-10, 1) # That is x = -10 and y = 1
steps = 30
lr = 0.4 # High to provoke the oscillation

path_sgd = run_standard_sgd(*start_pos, lr, steps)
path_rms = run_rmsprop(*start_pos, lr, beta = 0.9, steps = steps)

# Plotting with the help of matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the paths
ax.plot(path_sgd[:, 0], path_sgd[:, 1], 'o-', color='blue', alpha=0.5, label='Standard SGD (Bouncy)')
ax.plot(path_rms[:, 0], path_rms[:, 1], 'o-', color='red', linewidth=2, label='RMSprop (Stable)')

# Add formatting
ax.set_title("SGD vs RMSprop on an Elongated Canyon", fontsize=14)
ax.set_xlabel("X (Flat Direction)")
ax.set_ylabel("Y (Steep Direction)")
ax.legend()
ax.grid(True, alpha=0.3)

# Drawing the center target
ax.scatter(0, 0, color='black', marker='x', s=100, label='Target (Minimum)')

plt.show()

