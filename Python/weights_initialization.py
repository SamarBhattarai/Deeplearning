import numpy as np

# We are performing this check to prove the math behind Xavier Initialization
# (which is what the formula np.sqrt(1/n) represents).
# np.random.randn produces numbers with a variance of 1.0.
# Multiplying a random variable by a constant k scales its variance by k^2.
# Here, the constant is sqrt{1/4}.Therefore, the new variance should be (sqrt{1/4})^2 = 1/4 = 0.25.

# Here, we initialize weights in such a way that the vanishing gradient or exploding gradient 
# problem is less to a greater extent.

weights = np.random.randn(50, 40) * np.sqrt(1/ 4) 
# Here, we are assuming a layer contains 50 units/neurons and 40 units or input nodes in previous layer.
# And note that np.random.randn produces number with a variance of 1.0

print(weights)
# Now finding out the variance of weights
variance = np.var(weights)
print(f"Variance of weights: {variance}")