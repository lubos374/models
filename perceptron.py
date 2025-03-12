import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.4, epochs=999):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias term
        return self.activation(np.dot(self.weights, x))
    
    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Add bias term
                y_pred = self.predict(X[i])
                error = y[i] - y_pred
                self.weights += self.learning_rate * error * x_i
                total_error += abs(error)

            # Print total error each epoch
            print(f"Epoch {epoch+1}/{self.epochs}, Total Error: {total_error}")

if __name__ == "__main__":
    import pandas as pd

    # Load the CSV
    data = pd.read_csv("complex_student_scores.csv")

    # Separate features and labels (last column is the label)
    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values

    print("Loaded data shape:", X_train.shape, "Labels shape:", y_train.shape)

    # Create Perceptron with correct input_size
    input_size = X_train.shape[1]
    perceptron = Perceptron(input_size=input_size, learning_rate=0.01, epochs=999)

    # Train with CSV data
    perceptron.train(X_train, y_train)
