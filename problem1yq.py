import numpy as np
from efficient_cancer_data import read_training_data

'''
PROBLEM 1

Test:
Incorrect classifications on training data: 0.00%
Sucess Rate of Training: 63.67%

OUTPUT:
Incorrect classifications on training data: 4.67%
Success rate on training data: 95.33%
Incorrect classifications on validation data: 3.08%
Success rate on validation data: 96.92%

What is the percentage of samples that are incorrectly classified?
    - 3.08% Validation & 
    - 4.67% Training
Is it greater or smaller than the success rate on the training data
    - Validation data is GREATER than the training data
'''
# Load training & validation data
A, b = read_training_data('train.data')
A_val, b_val = read_training_data('validate.data')

# Convert to numpy arrays:
    #.flattan() is used to convert a 2D array to 1D array
    # For me, doing it because I'm getting a shape error, I think it was making matrix
    # and I needed a vector to calculate

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
'''
# Predict on validation data
predictions = A_val @ x
classified_predictions = np.where(predictions >= 0, 1, -1)

# Calculate
incorrect_classifications = np.mean(classified_predictions != b_val) * 100
print(f"Percentage of incorrect classifications: {incorrect_classifications:.2f}%")

success_rate_val = 100 - incorrect_classifications
print(f"Success rate on validation data: {success_rate_val:.2f}%")
'''
# Predict on training data
train_predictions = A @ x
train_classified = np.where(train_predictions >= 0, 1, -1)
train_incorrect_classifications = np.mean(train_classified != b) * 100
train_success_rate = 100 - train_incorrect_classifications

# Predict on validation data
val_predictions = A_val @ x
val_classified = np.where(val_predictions >= 0, 1, -1)
val_incorrect_classifications = np.mean(val_classified != b_val) * 100
val_success_rate = 100 - val_incorrect_classifications

# Display results
print(f"Percentage of incorrect classifications on training data: {train_incorrect_classifications:.2f}%")
print(f"Success rate on training data: {train_success_rate:.2f}%")
print(f"Percentage of incorrect classifications on validation data: {val_incorrect_classifications:.2f}%")
print(f"Success rate on validation data: {val_success_rate:.2f}%")