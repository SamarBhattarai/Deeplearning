import numpy as np
import tensorflow as tf

# Initialize the weight
w = tf.Variable(0, dtype=tf.float32)

# Define both optimizers
optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.1)
optimizer_2 = tf.keras.optimizers.SGD(learning_rate=0.1)

def training(optimizer_choice="Adam"):
    with tf.GradientTape() as tape:
        cost = w ** 2 - 10 * w + 25
        
    trainable_parameters = [w]
    grads = tape.gradient(cost, trainable_parameters)
    
    # Fix 1: Use '==' for comparison
    if optimizer_choice == "Adam":
        optimizer_1.apply_gradients(zip(grads, trainable_parameters))
    elif optimizer_choice == "SGD":
        optimizer_2.apply_gradients(zip(grads, trainable_parameters))

# --- Test 1: Adam ---
print("Training with Adam (1000 steps)...")
for i in range(1000):
    training(optimizer_choice="Adam")
    
# .numpy() extracts the clean float value from the tensor
print(f"w is now: {w.numpy():.4f}\n") 

# Fix 2: Reset 'w' back to 0 before testing SGD!
w.assign(0.0)

# --- Test 2: SGD ---
print("Training with SGD (20 steps)...")
for i in range(20):
    training(optimizer_choice="SGD")
    
print(f"w is now: {w.numpy():.4f}")