import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Compute cost function
def compute_cost(X, y, theta):
  m = len(y)
  predictions = X.dot(theta)
  cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
  return cost


# Minimize cost function
def gradient_descent(X, y, theta, learning_rate, iterations):
  m = len(y)
  cost_history = np.zeros(iterations)

  for it in range(iterations):
    prediction = np.dot(X, theta)
    theta = theta - (1 / m) * learning_rate * (X.T.dot(prediction - y))
    cost_history[it] = compute_cost(X, y, theta)

  return theta, cost_history


# Make predictions
def predict(X, theta):
  return X.dot(theta)


# Find theta directly
def normal_equations(X, y):
  return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


# Adjusted for CSV
file_path = 'boston.csv'

# Load dataset
df = pd.read_csv(file_path)

# Split dataset
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add intercept term
X_train_scaled = np.concatenate(
    [np.ones((X_train_scaled.shape[0], 1)), X_train_scaled], axis=1)
X_test_scaled = np.concatenate(
    [np.ones((X_test_scaled.shape[0], 1)), X_test_scaled], axis=1)

# Set initial parameters
theta_initial = np.zeros(X_train_scaled.shape[1])
learning_rate = 0.01
iterations = 1000

# Apply gradient descent
theta, cost_history = gradient_descent(X_train_scaled, y_train, theta_initial,
                                       learning_rate, iterations)

# Predict MEDV values
predictions = predict(X_test_scaled, theta)

# Calculate MSE
mse = np.mean((predictions - y_test)**2)

# Normal Equations alternative
theta_normal_eq = normal_equations(X_train_scaled, y_train)
predictions_normal_eq = predict(X_test_scaled, theta_normal_eq)
mse_normal_eq = np.mean((predictions_normal_eq - y_test)**2)

# Writing results to a file
output_file = "results.txt"
with open(output_file, "w") as f:
  f.write(f"Final Theta Values (Gradient Descent): {theta}\n")
  f.write(f"Mean Squared Error on Test Set (Gradient Descent): {mse}\n")
  f.write(f"Final Theta Values (Normal Equations): {theta_normal_eq}\n")
  f.write(
      f"Mean Squared Error on Test Set (Normal Equations): {mse_normal_eq}\n")

# Display the results
print(f"Results saved to {output_file}")
