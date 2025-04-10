# ai
computers mimicking human intelligence (perception,learning, reasoning,decision ). applications include robots and self-driving cars.
```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some sample data: X (feature) and Y (target)
# X represents the input feature (e.g., hours of study)
# Y represents the output target (e.g., exam scores)
X = np.array([[1], [2], [3], [4], [5]])  # Input: Hours of study
Y = np.array([1, 2, 2.8, 4.1, 5.2])  # Output: Exam scores

# Create a linear regression model
model = LinearRegression()

# Train the model (i.e., fit it to the data)
model.fit(X, Y)

# Make predictions using the trained model
predictions = model.predict(X)

# Plotting the data and the regression line
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Hours of Study')
plt.ylabel('Exam Score')
plt.title('Simple Linear Regression: Study Hours vs Exam Score')
plt.legend()
plt.show()

# Print the coefficients (slope) and intercept of the regression line
print(f"Coefficient (slope): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

```

# machine learning
use data to find a function, which make an input x predict an output y. 
keys: 
data: our obervations (x,y)
model: the function we wanna to find f(x)
learning: the process to find the optimal weights and parameters
inference: to use learned f to predict new x and get the result. 


```python
import random

# Generate training data: y = 2x + 1 with some noise
X = [i for i in range(10)]  # Input values
Y = [2 * x + 1 + random.uniform(-0.5, 0.5) for x in X]  # Output values with noise

# Initialize model parameters (weights)
w = random.random()  # slope
b = random.random()  # intercept

# Learning rate
lr = 0.01

# Train using simple gradient descent
for epoch in range(100):
    total_loss = 0
    for x, y in zip(X, Y):
        y_pred = w * x + b  # predicted value

        # Compute loss (mean squared error)
        loss = (y - y_pred) ** 2
        total_loss += loss

        # Compute gradients
        dw = -2 * x * (y - y_pred)
        db = -2 * (y - y_pred)

        # Update weights
        w -= lr * dw
        b -= lr * db

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, w={w:.4f}, b={b:.4f}")

# inference
test_x = 20
test_y_pred = w * test_x + b
print(f"\nPrediction: if x = {test_x}, then y ≈ {test_y_pred:.2f}")
````


# AutoML
automatic the process from develop and deploy an ML model

```python
import autosklearn.classification
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load dataset
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42
)

# Step 2: AutoML - Automatically search and train models
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,      # total search time in seconds
    per_run_time_limit=30,           # time per model trial
    memory_limit=1024                # limit memory usage (MB)
)

automl.fit(X_train, y_train)

# Step 3: Predict on test data
y_pred = automl.predict(X_test)

# Step 4: Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 5: Save trained model for deployment
joblib.dump(automl, "best_model.pkl")
```