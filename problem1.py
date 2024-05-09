

from efficient_cancer_data import read_training_data
from numpy.linalg import qr 
import numpy as np

# 1a
A, b = read_training_data("train.data")

"""
print(A.shape) #(300,30)
print(b)
print(A.transpose())
"""

# QR factorization
q,r = qr(A);
"""
print('Q:',q)
print('R:',r)
"""
#np.linalg.inv(r)
#q.transpose()


# least-squares solution with QR factorization
Qb = np.matmul(q.transpose(),b)
X = np.matmul(np.linalg.inv(r), Qb)

"""
b = np.dot(q,r)
print('QR:',b)
print('X:',X)
"""


#1b

VA, Vb = read_training_data("validate.data")
#print(VA.shape) #(260,30)
PVA = np.matmul(VA,X)
for index in range(len(PVA)):
    if(PVA[index]>=0):
        PVA[index] = 1
    else:
        PVA[index] = -1


count = 0 
totalCount = len(PVA)   
for i in range(len(PVA)):
    if(PVA[i] == Vb[i]):
        count = count + 1
        
#print(len(PA))
#print(len(Vb))
#print(PA)
#print(b)


#print(np.mean(PA == Vb))
#print(count)

#1c    

incorrectly = (totalCount - count)/totalCount * 100
correctly = count/totalCount * 100
print("percentage of samples that are correctly classified (validate.data): ",correctly,"%")
print("percentage of samples that are incorrectly classified (validate.data): ",incorrectly,"%")

PTA = np.matmul(A,X)
for position in range(len(PTA)):
    if(PTA[position]>=0):
        PTA[position] = 1
    else:
        PTA[position] = -1


count2 = 0 
totalCount2 = len(PTA)   
for j in range(len(PTA)):
    if(PTA[j] == b[j]):
        count2 = count2 + 1

        
incorrectly2 = (totalCount2 - count2)/totalCount2 * 100
correctly2 = count2/totalCount2 * 100  
print("percentage of samples that are correctly classified (train.data): ",correctly2,"%")
print("percentage of samples that are incorrectly classified (train.data): ",incorrectly2,"%")

print("Is it greater than the success rate on the training data:",correctly>correctly2)
