import numpy as np
from sympy import Matrix
from efficient_cancer_data import read_training_data

'''
First Run:
Accuracy on validation data: 63.67%
Accuracy on training data: 0.00% (Literally the model isn't even looking at the training data...)
'''

def qr_least_squares(A, b):
    A_np = np.array(A).astype(np.float64)
    b_np = np.array(b).astype(np.float64).flatten()
    
    # QR decomposition of A
    Q, R = np.linalg.qr(A_np)
    # Solve R*x = Q.T * b for x
    x = np.linalg.solve(R, Q.T @ b_np)
    return x

def predict(A, x):
    # Use linear model to predict
    A_np = np.array(A).astype(np.float64)
    predictions = A_np @ x
    # Classify predictions 
    predicted_classes = np.where(predictions >= 0, 1, -1)
    return predicted_classes

def evaluate_predictions(true_labels, predicted_labels):
    # Calculate the accuracy of the predictions
    accuracy = np.mean(true_labels == predicted_labels)
    return accuracy

# Load training & validation data
A, b = read_training_data('train.data')
A_val, b_val = read_training_data('validate.data')

# Compute least squares solution using QR decomposition
x = qr_least_squares(A, b)

# Predict on validation data
predicted_classes = predict(A_val, x)

# Accuracy on Validation and Training data
accuracy = evaluate_predictions(np.array(b_val).astype(int), predicted_classes)
print(f"Accuracy on validation data: {accuracy*100:.2f}%")

predicted_classes_train = predict(A, x)
accuracy_train = evaluate_predictions(b, predicted_classes_train)
print(f"Accuracy on training data: {accuracy_train*100:.2f}%")