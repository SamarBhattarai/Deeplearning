import numpy as np

# 1. SETUP DATA (Corrected Syntax)
# We have 3 scales of gradients:
# - Tiny (0.01, 0.1)
# - Medium (2, 3)
# - Huge (100, 10)
dW = np.array([
    [0.1, 2.0, 10.0], 
    [10.0, 0.2, 3.0], 
    [100.0, 0.01, 7.0]
])

# 2. HYPERPARAMETERS
s_x = np.zeros(shape = dW.shape) # Initialize cache to zeros
beta = 0.9              # Standard friction
learning_rate = 0.01    # The "Global" learning rate
epsilon = 1e-8          # Safety for division

# 3. THE RMSPROP UPDATE (Simulating 1 Step)
s_x = (beta * s_x) + (1 - beta) * np.square(dW)

# 4. CALCULATE THE "EFFECTIVE" STEP
# This is the actual value we subtract from the weights: alpha * dW / sqrt(S)
effective_step = learning_rate * dW / (np.sqrt(s_x) + epsilon)

# 5. VISUALIZE THE "EQUALIZING" EFFECT
print(f"{'Original Gradient (dW)':<25} | {'RMSprop Step Size':<20}")
print("-" * 50)

# Flatten arrays to print them side-by-side
flat_dW = dW.flatten()
flat_step = effective_step.flatten()

for original, new_step in zip(flat_dW, flat_step):
    print(f"{original:<25.4f} | {new_step:<20.6f}")

print("-" * 50)
print("\nOBSERVATION:")
print("1. The HUGE gradient (100.0) became small (0.0316).")
print("2. The TINY gradient (0.01) ALSO became small (0.0316).")
print("RMSprop forced them to move at the SAME SPEED!")