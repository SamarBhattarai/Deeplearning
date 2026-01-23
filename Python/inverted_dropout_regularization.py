import numpy as np

np.random.seed(87)
A = np.random.randn(5, 4)
# This creates a matrix representing activations 
# from a hidden layer with 5 units across 4 training examples.
print(A)

keep_prob = 0.8
D = np.random.rand(A.shape[0], A.shape[1]) 
print(D)
D = D < keep_prob
# Creating a random mask D
print(D)

A = np.multiply(A, D)
# Applying the mask
print(A)

A = A/ keep_prob
print(A)
# Scale Activations (The "Inverted" Step)
# Dividing by keep_prob scales the remaining active neurons back up.
# If you keep 80% of neurons, you boost their signal by 
# \(1/0.8=1.25\) to maintain the same expected value as if no neurons were dropped.