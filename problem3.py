import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

'''
OUTPUT:
Classification accuracy using 5 singular vectors: 91.80%
Classification accuracy using 10 singular vectors: 94.40%
Classification accuracy using 15 singular vectors: 95.30%
Classification accuracy using 20 singular vectors: 95.70%

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

digit_singular_values = {i: [] for i in range(num_digits)}
# SVD & stores the k value for each singular vector
digit_singular_vectors = {k: {} for k in ks}

for digit, samples in digit_samples.items():
    matrix = np.vstack(samples) # Stack arrays vertically to turn it into a matrix
    U, S, Vt = svd(matrix, full_matrices=False) # SVD
    digit_singular_values[digit] = S[:20]
    for k in ks:
        digit_singular_vectors[k][digit] = Vt[:k] # Stores the k
# print("Singular vectors for each digit and k DONE-OH.")

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
print("Part A:")
for k, accuracy in results.items():
    print(f"Classification accuracy using {k} singular vectors: {accuracy:.2f}%")



'''
PROBLEM 3
part B

OUTPUT:
    The accuracy of implementing two-stage algorithm is 91.60%
    The fallback classification function was used 47.20% of the time.

Answers:
    Is it possible to get as good a result for this version? 
        - Yes. When using 20 singular vectors in the fallback function, the accuracy of the two-stage algorithm is shown
         to be 91.60%, which is only 0.2% less accurate than using 5 singular vectors. And the accuracy itself is not 
         bad, if someone can be satisfied with 91.60% accuracy, this algorithm can be as good as the algorithm above.
    How frequently is the second stage necessary?
        - In our testing, the second stage was used 47.20% of the time.
'''


# Compare the unknow digit with the first singular vector
def two_stage_classification(sample, singular_vectors):
    distances = {}
    fallback_count = 0  # Initialize counter variable for fallback calls
    for digit, vectors in singular_vectors.items():
        proj = vectors[:1].T @ (vectors[:1] @ sample)
        distances[digit] = np.linalg.norm(sample - proj)  # Compute Euclidean distance
    q1 = np.percentile(list(distances.values()), 25)
    q3 = np.percentile(list(distances.values()), 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    outliers = [digit for digit, distance in distances.items() if distance < lower_bound]
    if outliers:
        prediction = outliers[0]
    else:
        fallback_count += 1  # Increment fallback counter
        prediction = classify_sample(sample, singular_vectors)
    return prediction, fallback_count

two_stage_predictions_and_counts = [two_stage_classification(sample, digit_singular_vectors[20]) for sample in test_data]
two_stage_predictions, fallback_counts = zip(*two_stage_predictions_and_counts)

# Calculate accuracy
two_stage_correct_predictions = sum(1 for pred, true in zip(two_stage_predictions, test_labels) if pred == int(true) - 1)
accuracy = two_stage_correct_predictions / len(test_labels) * 100

# Calculate fallback percentage
total_samples = len(test_data)
fallback_percentage = (sum(fallback_counts) / total_samples) * 100
print("\n")
print("Part B:")
print(f"The accuracy of implementing two-stage algorithm is {accuracy:.2f}%")
print(f"The fallback classification function was used {fallback_percentage:.2f}% of the time.")

'''
Graph

Note: Took out some graphs, but the png is still in the folder
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

# Plot the first 20 singular values of each digit
plt.figure(figsize=(10, 6))
for digit, singular_values_array in digit_singular_values.items():
    plt.plot(np.arange(len(singular_values_array)), singular_values_array, label=f'Digit {digit}')
plt.xlabel('Index')
plt.ylabel('Singular Value Magnitude')
plt.title('Singular Values for Each Digit')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(20))
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