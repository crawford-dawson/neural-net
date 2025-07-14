# Working of this tutoriaL: https://medium.com/data-science/first-neural-network-for-beginners-explained-with-code-4cfd37e06eaf
import numpy as np
import random
import os

lr = 1 # learning rate
bias = 1 # bias
weights = [random.random(), random.random(), random.random()] # weights for 3 random inputs on two neurons

def Perceptron(input1, input2, output):
    # weighted sum
    weighted_sum = (input1 * weights[0]) + (input2 * weights[1]) + (bias * weights[2])
    # activation function
    if weighted_sum > 0:
        return 1
    else:
        return 0
    # tutorial does this a little differently, but this is more intuitive (at least to me)
    error = output - weighted_sum
    # update weights
    weights[0] += lr * error * input1
    weights[1] += lr * error * input2
    weights[2] += lr * error * bias

for i in range(1000): # training the perceptron
    # True False values for OR gate
    Perceptron(1, 1, 1) # training with input (1, 1) expecting output 1
    Perceptron(1, 0, 1) # training with input (1, 0) expecting output 1
    Perceptron(0, 1, 1) # training with input (0, 1) expecting output 1
    Perceptron(0, 0 , 0) # training with input (0, 0) expecting output 0

x = int(input("Enter first input (0 or 1): "))
y = int(input("Enter second input (0 or 1): "))
outputP = (x * weights[0]) + (y * weights[1]) + (bias * weights[2])
if outputP > 0: # activation function
    outputP = 1
else:
    outputP = 0
print(f"Output for inputs ({x}, {y}): {outputP}")