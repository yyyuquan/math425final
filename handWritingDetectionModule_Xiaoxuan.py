import numpy as np

# Construct a matrix for all handwritings
# row 0 - 399 for digit 0, labeled as 10 in label.txt
# row 400 - 799 for digit 1

def read_matrix_from_txt(path, num_examples_per_digit):
    data = np.loadtxt(path, delimiter=",")

    # Reshape the data into a 3D array with shape (num_examples, 20, 20)
    data_reshaped = data.reshape(-1, 20, 20)

    # A dictionary to store matrices for each digit
    digit_matrices = {}

    for digit in range(10):
        # Extract the examples for the current digit class
        examples = data_reshaped[digit * num_examples_per_digit: (digit + 1) * num_examples_per_digit]
        digit_matrices[digit] = examples.reshape(num_examples_per_digit, -1)

    return digit_matrices

class handWritingDetection:
    def __init__(self, path, k, s):
        self.k = k # Number of images for every digit
        self.s = s  # Number of singular vectors as basis
        self.digit_matrices = read_matrix_from_txt(path, k)
        self.mu = {} # Mean
        self.V = {}
        self.Fhat = {} # Saved data in smaller dimension

        # Compute mean and SVD representation for each digit class
        for digit, matrix in self.digit_matrices.items():
            self.mu[digit] = np.mean(matrix, axis=0)
            self.fbar = (matrix - self.mu[digit]).T
            _, _, self.V[digit] = np.linalg.svd(matrix)
            # self.s[digit] = 0
            # Compute low-rank representation for each digit class
            self.Fhat[digit] = self.lowrankrep(self.s, self.fbar, digit)

    def lowrankrep(self, s, f, digit):
        return (self.V[digit][:, :s].T @ f)

    def detectWriting(self, g):
        min_norm = float('inf')
        predicted_digit = None
        for digit in range(10):
            # Project unknown image onto low-dimensional space for each digit class
            ghat = self.lowrankrep(self.s, np.reshape(g - self.mu[digit], (len(g), 1)), digit)
            # Calculate the norm between the projected image and the Fhat for the digit class
            norms = np.linalg.norm(self.Fhat[digit] - ghat, axis=0)
            min_index = np.argmin(norms)
            norm = norms[min_index]
            # Check if the norm is smaller than the minimum norm found so far
            if norm < min_norm:
                min_norm = norm
                predicted_digit = digit
        return predicted_digit

