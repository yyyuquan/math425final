import numpy as np
from efficient_cancer_data import read_training_data

A, b = read_training_data('train.data')
A = np.array(A, dtype=float)
b = np.array(b, dtype=float)
Q, R = np.linalg.qr(A)
x = np.linalg.solve(R, Q.T @ b)
print(x)

