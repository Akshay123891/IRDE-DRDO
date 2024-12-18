import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
# Load data
#X = np.loadtxt('logisticX.csv', delimiter=',')
#y = np.loadtxt('logisticY.csv', delimiter=',')
X=pd.read_csv('C:/Users/HTC/Downloads/logisticX.csv').values
y=pd.read_csv('C:/Users/HTC/Downloads/logisticY.csv').values.flatten()
# Add intercept term to X
X = np.c_[np.ones(X.shape[0]), X]

# Initialize theta
theta = np.zeros(X.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Newton's method
def newtons_method(X, y, theta, iterations=10):
    for _ in range(iterations):
        # Compute the predictions
        h = sigmoid(X @ theta)
        
        # Gradient of the log-likelihood
        gradient = X.T @ (h - y) 
        
        # Hessian matrix
        R = np.diag(h * (1 - h))
        H = X.T @ R @ X
        # Update theta
        theta = theta - np.linalg.inv(H) @ gradient

    return theta

#check dimensions 
print("X shape:", X.shape)
print("y shape:", y.shape)
print("theta shape:", theta.shape)

# Run Newton's method
theta_optimized = newtons_method(X, y, theta)

print("Optimized coefficients:", theta_optimized)

# Compute predictions 
predicted_probabilities = sigmoid(X @ theta_optimized)

#Convert probabilitied to binary (0 or 1)
predicted_labels = (predicted_probabilities >=0.5).astype(int)

#Compute accuracy 
accuracy = np.mean(predicted_labels == y)
print(f"Accuracy: {accuracy:.4f}")

conf_matrix = confusion_matrix(y, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)