import numpy as np
import sympy as sp
from efficient_cancer_data import read_training_data

# Problem 3
def load_data(filepath):
    return np.loadtxt(filepath)

train_data = load_data('handwriting_training_set.txt')
train_label = load_data('handwriting_training_set_labels.txt')

test_data = load_data('handwriting_training_set.txt')
test_labels = load_data('handwriting_training_set_labels.txt')

