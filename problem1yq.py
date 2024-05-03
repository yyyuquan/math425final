import numpy as np
from efficient_cancer_data import read_training_data

# Load training data and validation data
A, b = read_training_data('train.data')
A_val, b_val = read_training_data('validate.data')

# Convert to numpy arrays
#.flattan() is used to convert a 2D array to 1D array

A = np.array(A).astype(np.float64)
b = np.array(b).astype(np.float64).flatten()
A_val = np.array(A_val).astype(np.float64)
b_val = np.array(b_val).astype(np.float64).flatten()

#print("Shape of A:", A.shape) # 300,30
#print("Shape of b:", b.shape) # 300,1
#print("Shape of A_val:", A_val.shape) #260,30
#print("Shape of b_val:", b_val.shape) #260,1?

# QR
Q, R = np.linalg.qr(A)
x = np.linalg.solve(R, Q.T @ b)
#print("Coefficients:", x)

# Predict on validation data
predictions = A_val @ x
classified_predictions = np.where(predictions >= 0, 1, -1)

# Calculate
incorrect_classifications = np.mean(classified_predictions != b_val) * 100
print(f"Percentage of incorrect classifications: {incorrect_classifications}%") #3.08%

success_rate_val = 100 - incorrect_classifications
print(f"Success rate on validation data: {success_rate_val}%") #96.92%