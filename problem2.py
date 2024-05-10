from efficient_cancer_data import read_training_data
from sympy import *

# part (a)

A, b = read_training_data("train.data")
A.shape # dimension of the matrix A = (300, 30)
b.shape # dimension of the vector b = (300, 1)
half_weighted_indices = [1, 7, 25, 26, 29]
W = zeros(30, 30)
for i in range(30):
    if i in half_weighted_indices:
        W[i, i] = 1
    else:
        W[i, i] = 2
WA = A * W # the matrix A after being weighted
Q, R = WA.QRdecomposition()
x = R.solve(Transpose(Q) * b) # least square solution x for WAx = (QR)x = b where Q is orthogonal
""" (Alternatively)
The equation is (WA)x = b with normal equation (WA)T WA x = (WA)T b
WATWA = Transpose(WA) * WA
x = WATWA.solve(Transpose(WA) * b)
"""

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
val_cor_perc = (val_count / 260) * 100 # the percentage of samples that match
val_inc_perc = 100 - val_cor_perc # the incorrect percentange of samples that dismatch = 
print(val_inc_perc)

#finding the predicted vector for training data
pred_vec = []
for i in range(300):
    pred = A.row(i) * x
    if (pred[0] >= 0):
        pred_vec.append(1)
    else:
        pred_vec.append(-1)

count = 0
# find the number of samples that match
for i in range(300):
    if (pred_vec[i] == b[i]):
        count = count + 1
cor_perc = (count / 300) * 100 # the percentage of samples that match
inc_perc = 100 - cor_perc # the incorrect percentange of samples that dismatch = 
print(inc_perc)

#The values of inc_perc for both training data and validate data is always different each time I run the code
#There are sometimes were inc_perc of training data is greater than inc_perc of validata data and sometime it is less than.