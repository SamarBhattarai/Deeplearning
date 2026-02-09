import numpy as np

def l2_regularization(W, lambd, m):
    """
    l2_regularization
    :param W: Weight matrix
    :param lambd: regularization parameter e.g 0.7
    :param m: number of training examples

    """
    # 1. Effect on COST (Forward Prop)
    # We add a "penalty" term to the cost
    l2_cost_penalty = (lambd / (2 * m)) * np.sum(np.square(W))

    # 2. Effect on GRADIENTS (Back Prop)
    # We add a term that pushes W towards zero
    # This is why it's called "Weight Decay"
    l2_gradient_penalty = (lambd / m) * W

    return l2_cost_penalty, l2_gradient_penalty

# Example Usage
W = np.array([[10, 0.5, -3]]) # Some large weights
m = 1000
lambd = 0.9

penalty, dW_reg = l2_regularization(W, lambd, m)

print(f"Original Weights: {W}")
print(f"Gradient Push (L2): {dW_reg}") 
# Notice: The push is proportional to the weight size. 
# 10 gets a big push, 0.5 gets a tiny push.

def l1_regularization(W, lambd, m):
    # 1. Effect on COST
    l1_cost_penalty = (lambd / (2 * m)) * np.sum(np.abs(W))
    
    # 2. Effect on GRADIENTS
    # np.sign(W) returns 1 if W is positive, -1 if negative, 0 if 0.
    # It pushes constants towards zero at a constant rate, regardless of size.
    l1_gradient_penalty = (lambd / m) * np.sign(W)
    
    return l1_cost_penalty, l1_gradient_penalty

# Example Usage
W = np.array([[10, 0.5, -3]]) 
m = 1000
lambd = 0.9

penalty, dW_reg = l1_regularization(W, lambd, m)

print(f"Original Weights: {W}")
print(f"Gradient Push (L1): {dW_reg}")
# Notice: The push is CONSTANT (0.0009 or -0.0009).
# It punishes small weights (0.5) just as hard as big weights (10).
# This eventually forces 0.5 to hit exactly 0.