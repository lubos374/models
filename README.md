# One-Layer Perceptron for Student Success Classification

A simple implementation of a single-layer perceptron neural network that learns to classify students as successful (1) or unsuccessful (0) based on various factors including study habits and lifestyle.

## Overview

This project demonstrates how a basic perceptron can learn to classify students into two categories based on five input features:
- Study Hours
- Sleep Hours
- Exercise Hours
- Job Hours (work commitments)
- Social Hours

The model learns the weights for each factor to make accurate predictions about student success.

## Project Files

- `perceptron.py` - The main Python implementation of the perceptron algorithm
- `complex_student_scores.csv` - Dataset containing student factors and success labels

## How It Works

The perceptron performs binary classification through these steps:

1. Initializes random weights for each input feature (plus a bias term)
2. For each training example:
   - Calculates the weighted sum of inputs
   - Applies a step activation function (returns 1 if value ≥ 0, otherwise 0)
   - Updates weights based on the prediction error
3. Repeats the process for a specified number of epochs
4. Uses the trained weights to make predictions on new data

## Implementation Details

The perceptron model includes:

- Configurable learning rate (default: 0.4)
- Configurable number of training epochs (default: 999)
- Automatic handling of bias term
- Error tracking during training

## Requirements

- Python 3.x
- NumPy
- Pandas (for data loading)

```
pip install numpy pandas
```

## Usage

### Training the Model

```python
import pandas as pd
from perceptron import Perceptron

# Load data
data = pd.read_csv("complex_student_scores.csv")
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels

# Create and train the perceptron
model = Perceptron(input_size=X.shape[1], learning_rate=0.01, epochs=999)
model.train(X, y)
```

### Making Predictions

```python
# Example student data: [Study_Hours, Sleep_Hours, Exercise_Hours, Job_Hours, Social_Hours]
new_student = [7, 8, 2, 1, 2]

# Predict if the student will be successful
prediction = model.predict(new_student)
print("Predicted outcome:", "Successful" if prediction == 1 else "Unsuccessful")
```

## Dataset Information

The dataset includes 100 student records with these features:

| Feature | Description | Range |
|---------|-------------|-------|
| Study_Hours | Hours spent studying daily | 0-10 |
| Sleep_Hours | Hours of sleep per night | 4-9 |
| Exercise_Hours | Hours of physical activity | 0-3 |
| Job_Hours | Hours spent at work | 0-5 |
| Social_Hours | Hours spent socializing | 0-4 |
| Label | Success indicator (1=success, 0=not success) | 0 or 1 |

## Model Performance

The perceptron gradually reduces error over training epochs and can identify patterns in student success. The final accuracy depends on:

- The learning rate
- Number of training epochs
- Quality and quantity of training data
- Inherent separability of the data

## Limitations

- As a single-layer perceptron, this model can only solve linearly separable problems
- The binary step activation function may not capture nuanced relationships
- No regularization is implemented to prevent overfitting

## Future Improvements

Potential enhancements to consider:

1. Add data normalization to improve training stability
2. Implement cross-validation to assess model performance
3. Extend to a multi-layer perceptron for handling non-linear boundaries
4. Add visualization tools for the decision boundary
5. Implement different activation functions (sigmoid, ReLU)

## License

MIT License
