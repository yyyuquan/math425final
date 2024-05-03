import numpy as np

'''
PROBLEM 3
'''
# Load all the data
def load_data(filepath):
    return np.loadtxt(filepath, delimiter=',')

training_data = load_data('handwriting_training_set.txt')
training_labels = load_data('handwriting_training_set_labels.txt')
test_data = load_data('handwriting_test_set.txt')
test_labels = load_data('handwriting_test_set_labels.txt')

#print("Training data shape:", training_data.shape)  #4000,400
#print("Training labels shape:", training_labels.shape)  #4000,1
#print("Test data shape:", test_data.shape)  #1000,400
#print("Test labels shape:", test_labels.shape)  #1000,1

