from efficient_cancer_data import read_training_data
from sympy import *

# part (a)

A, b = read_training_data("train.data")
A.shape # dimension of the matrix A = (300, 30)
b.shape # dimension of the vector b = (300, 1)
Q, R = A.QRdecomposition()
Q.shape # dimension of the matrix Q = (300, 31)
R.shape # dimension of the matrix R = (31, 31)
x = R.solve(Transpose(Q) * b) # least square solution x for Ax = (QR)x = b where Q is orthogonal
x.shape # dimension of the vector B = (31, 1)

# part (b)

val_A, val_b = read_training_data('validate.data')
val_A.shape # dimension of the matrix val_A = (260, 30)
val_b.shape # dimension of the vector val_b = (260, 1)
val_pred_vec = []
for i in range(260):
    val_pred = val_A.row(i) * x
    if (val_pred[0] >= 0):
        val_pred_vec.append(1)
    else:
        val_pred_vec.append(-1)
len(val_pred_vec) # size of the list val_perd_vec = 260

# part (c)

val_count = 0
# find the number of samples that match
for i in range(260):
    if (val_pred_vec[i] == val_b[i]):
        val_count = val_count + 1
cor_perc = (val_count / 260) * 100 # the percentage of samples that match
inc_perc = 100 - cor_perc # the incorrect percentange of samples that dismatch = 3.07692307692308%
