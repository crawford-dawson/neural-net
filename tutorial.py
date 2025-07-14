# Working off of this tutorial: https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc

import numpy as np

n = [2, 3, 3, 1]
print("layer 0 / input layer size", n[0])
print("layer 1 size", n[1])
print("layer 2 size", n[2])
print("layer 3 size", n[3])

# Initialize weights and biases for a 3-layer neural network
W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)

# Print the shapes of the weights and biases
print("Weights for layer 1 shape:", W1.shape)
print("Weights for layer 2 shape:", W2.shape)
print("Weights for layer 3 shape:", W3.shape)
print("bias for layer 1 shape:", b1.shape)
print("bias for layer 2 shape:", b2.shape)
print("bias for layer 3 shape:", b3.shape)

# Example of a 2D array with 10 rows and 2 columns
# This could represent features like height and weight of individuals
# Each row is a different individual, and each column is a different feature
X = np.array([
    [150, 70], 
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
])

print(X.shape) # prints (10, 2)

# Transpose the matrix to have features in columns
A0 = X.T
print("Transposed X shape:", A0.shape) # prints (2, 10)

# Example of a 1D array with 10 elements
# This could represent binary outputs for a classification task (i.e. heart disease presence)
y = np.array([
    0,  
    1, 
    1, 
    0,
    0,
    1,
    1,
    0,
    1,
    0
])
m = 10

# we need to reshape to a n^[3] x m matrix
Y = y.reshape(n[3], m)
Y.shape

def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))

m = 10

# layer 1 calculations

Z1 = W1 @ A0 + b1  # the @ means matrix multiplication

assert Z1.shape == (n[1], m) # just checking if shapes are good
A1 = sigmoid(Z1)

# layer 2 calculations
Z2 = W2 @ A1 + b2
assert Z2.shape == (n[2], m)
A2 = sigmoid(Z2)

# layer 3 calculations
Z3 = W3 @ A2 + b3
assert Z3.shape == (n[3], m)
A3 = sigmoid(Z3)

print(A3.shape)  # prints (1, 10)
y_hat = A3

