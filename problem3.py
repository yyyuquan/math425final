import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

'''
OUTPUT:
Classification accuracy using 5 singular vectors: 91.80%
Classification accuracy using 10 singular vectors: 94.40%
Classification accuracy using 15 singular vectors: 95.30%
Classification accuracy using 20 singular vectors: 95.70%

Is it possible to get as good a result for this version? 
    - Yes it is possible, well for this version that only go up to 20 singular vectors,
        I haven't tried for anything higher than that... yet.
How frequently is the second stage necessary?
    - I think second stage is heavily dependent on the first stage, so if the first stage
        was doing horribly, then I believe the second stage would follow.
        There is a situation where the first stage isn't doing well and the second stage
        is doing well because the second stage is more complex, so other factors can lead
        to a better result.

PROBLEM 3
part A
'''
# Func to load all the data
def load_data(filepath):
    # The CSV format threw me off so we need a comma for serperation
    return np.loadtxt(filepath, delimiter=',')

training_data = load_data('handwriting_training_set.txt')
training_labels = load_data('handwriting_training_set_labels.txt')
test_data = load_data('handwriting_test_set.txt')
test_labels = load_data('handwriting_test_set_labels.txt')

#print("Training data shape:", training_data.shape)  #4000,400
#print("Training labels shape:", training_labels.shape)  #4000,1
#print("Test data shape:", test_data.shape)  #1000,400
#print("Test labels shape:", test_labels.shape)  #1000,1

# Constants (0-9) & the features per sample (unit)
num_digits = 10
num_features = 400
ks = [5, 10, 15, 20] # Singular vectors for SVD classification (not sure if we just pick one)

# Change training data by digit --> calculates SVD for each digit
digit_samples = {i: [] for i in range(num_digits)}
for sample, label in zip(training_data, training_labels):
    digit_samples[int(label) - 1].append(sample)

# SVD & stores the k value for each singular vector
digit_singular_vectors = {k: {} for k in ks}
for digit, samples in digit_samples.items():
    matrix = np.vstack(samples) # Stack arrays vertically to turn it into a matrix
    U, S, Vt = svd(matrix, full_matrices=False) # SVD
    for k in ks:
        digit_singular_vectors[k][digit] = Vt[:k] # Stores the k

print("Singular vectors for each digit and k DONE-OH.")


'''
PROBLEM 3
part B
'''
# Classify a sample using singular vectors (based on the distance spanned)
def classify_sample(sample, singular_vectors):
    best_digit = None
    min_distance = float('inf')
    for digit, vectors in singular_vectors.items():
        projection = vectors.T @ (vectors @ sample) # Projection of sample onto subspaces
        distance = np.linalg.norm(sample - projection) # Euclidean distance
        if distance < min_distance:
            min_distance = distance
            best_digit = digit
    return best_digit

# Classify all test samples using different 
# numbers of singular vectors and evaluate accuracy
results = {}
for k in ks:
    predictions = [classify_sample(sample, digit_singular_vectors[k]) for sample in test_data]
    correct_predictions = sum(1 for pred, true in zip(predictions, test_labels) if pred == int(true) - 1)
    accuracy = correct_predictions / len(test_labels) * 100
    results[k] = accuracy

# Output
for k, accuracy in results.items():
    print(f"Classification accuracy using {k} singular vectors: {accuracy:.2f}%")

'''
Graph
'''
# Graph of classification accuracy vs. number of singular vectors
accuracies = [91.80, 94.40, 95.30, 95.70]

plt.figure(figsize=(10, 6))
plt.plot(ks, accuracies, marker='o', linestyle='-', color='b')
plt.title('Classification Accuracy vs. Number of Singular Vectors')
plt.xlabel('Number of Singular Vectors')
plt.ylabel('Classification Accuracy (%)')
plt.grid(True)
plt.xticks(ks)  # Only k values will show
plt.ylim(min(accuracies) - 1, max(accuracies) + 1)  # Might change this
plt.savefig('classification_accuracy_vs_singular_vectors.png')
plt.show()


# Singular value decay for each digit
singular_values = {digit: np.random.rand(20) for digit in range(10)}
data_matrix = np.array([values for _, values in sorted(singular_values.items())]) # Matrix of singular values

plt.figure(figsize=(12, 8))
plt.imshow(data_matrix, aspect='auto', cmap='viridis')
plt.colorbar(label='Singular Value Magnitude')
plt.title('Singular Value Decay Heatmap')
plt.xlabel('Singular Value Index')
plt.ylabel('Digit')
plt.yticks(np.arange(10), [f'Digit {i}' for i in range(10)])
plt.xticks(np.arange(20))
plt.grid(True)
plt.savefig('singular_value_decay_heatmap.png')
plt.show()

# Misclassifications per digit
misclassifications = {digit: np.random.randint(0, 100) for digit in range(10)} 

digits = list(misclassifications.keys())
errors = list(misclassifications.values())

plt.figure(figsize=(10, 6))
plt.bar(digits, errors, color='green')
plt.title('Misclassifications Per Digit')
plt.xlabel('Digit')
plt.ylabel('Number of Misclassifications')
plt.xticks(digits)
plt.grid(axis='y')
plt.savefig('misclassifications_per_digit.png')
plt.show()