#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:04:44 2024

@author: henryboateng
"""

from efficient_cancer_data import read_training_data
import sympy as sp

#(a) find linear model
A, b = read_training_data("train.data")
# print(A.shape) #rows=300 cols=30
numRows = 300
Q, R = A.QRdecomposition()
invR = R.inv()
linear_model = invR * Q.transpose() * b
# print(linear_model) #answer to (a)

#(b) predict malignancy of tissue (malignancy if A * linear_model == 1) (benign if A * linear_model == -1) 
val_A, val_b = read_training_data('validate.data')
# print(val_A.shape) #rows=260 cols=30
val_numRows = 260

val_predictionVector = []
for i in range(val_numRows):
    val_predictionValue = val_A.row(i) * linear_model
    if (val_predictionValue[0] >= 0):
        val_predictionVector.append(1)
    else:
        val_predictionVector.append(-1)
# print(val_predictionVector) #answer to (b)

#(c) percentage of samples incorrectly classified, is it greater or smaller than success rate on training data
val_count = 0

for i in range(val_numRows):
    if (val_predictionVector[i] == val_b[i]):
        val_count += 1

incorrectly_classified_percent = (1 - (val_count/val_numRows)) * 100
print(incorrectly_classified_percent) #answer to (c) = 3.076923076923077

#finding success rate of training data
predictionVector = []
for i in range(numRows):
    predictionValue = A.row(i) * linear_model
    if (predictionValue[0] >= 0):
        predictionVector.append(1)
    else:
        predictionVector.append(-1)

count = 0

for i in range(numRows):
    if (predictionVector[i] == b[i]):
        count += 1

incorrectly_classified_percent = (1 - (count/numRows)) * 100
print(incorrectly_classified_percent) #4.666666666666663

#rest of the answer to (c)
print("the percentage of samples incorrectly classified from validate.data is less than")
print("the percentage of samples incorrectly classified from training data which means")
print("the success rate of the validate.data is greater than the success rate of the training data")