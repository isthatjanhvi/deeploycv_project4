import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical

# Task 1: Load and preprocess the Iris dataset
def load_and_preprocess_iris():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # Features: Sepal length, Sepal width, Petal length, Petal width
    y = iris.target  # Target: Species (0, 1, 2)

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # One-hot encode labels
    y = to_categorical(y, num_classes=3)

    # Reshape data for CNN input (4 features as 2x2 image)
    X = X.reshape(-1, 2, 2, 1)  # Shape: (samples, height, width, channels)

    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Task 2: Build a CNN model
def build_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (2, 2), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(1, 1)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 output neurons for 3 classes
    ])
    return model

# Task 3: Train and evaluate the model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=5, verbose=1)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

# Main function
if __name__ == "__main__":
    # Task 1: Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_iris()

    # Task 2: Build the CNN model
    model = build_cnn_model((2, 2, 1))  # Input shape: (2, 2, 1)

    # Task 3: Train and evaluate the model
    test_accuracy = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
