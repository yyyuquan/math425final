from handWritingDetectionModule_Xiaoxuan import handWritingDetection
import numpy as np
import matplotlib.pyplot as plt

def train_model(training_file, k, s):
    """
    Train the hand writing detection model with the specified parameters.

    Args:
    - training_file: Path to the training data file.
    - k: Number of examples for each digit in the training data.
    - s: Number of singular vectors as basis.

    Returns:
    - Trained hand writing detection model.
    """
    return handWritingDetection(training_file, k, s)

def test_model(model, testing_data, testing_labels):
    """
    Test the hand writing detection model on the testing dataset.

    Args:
    - model: Trained hand writing detection model.
    - testing_data: Testing data array.
    - testing_labels: Testing labels array.

    Returns:
    - Accuracy of the model on the testing dataset.
    """
    # Test the model on the test set
    correct_predictions = 0
    total_predictions = len(testing_data)

    for image, label in zip(testing_data, testing_labels):
        predicted_digit = model.detectWriting(image)
        if predicted_digit == label:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    return round(accuracy, 2)

def calculate_accuracy(correct_predictions, total_predictions):
    """
    Calculate accuracy given the number of correct predictions and total predictions.

    Args:
        correct_predictions (int): The number of correct predictions.
        total_predictions (int): The total number of predictions.

    Returns:
        float: The accuracy, or 0 if total_predictions is 0.
    """
    if total_predictions == 0:
        return 0
    else:
        return round((correct_predictions / total_predictions) * 100, 2)

def plot_accuracy_vs_basis_vectors(detectors, testing_data, testing_labels, s_values):
    """
    Plot the accuracy of the hand writing detection model as a function of the number of basis vectors.

    Args:
    - detectors: Dictionary containing detectors with different s values as keys.
    - testing_data: Testing data array.
    - testing_labels: Testing labels array.

    Returns:
    - None
    """
    accuracies = []

    # Iterate over the detectors and test each one on the testing dataset
    for s, detector in detectors.items():
        accuracy = test_model(detector, testing_data, testing_labels)
        accuracies.append(accuracy)

    # Plot accuracy vs. number of basis vectors
    plt.plot(s_values, accuracies, marker='o')
    plt.xlabel('Number of Basis Vectors (s)')
    plt.ylabel('Accuracy %')
    plt.title('Accuracy vs. Number of Basis Vectors')
    plt.grid(True)
    plt.xticks(s_values)
    plt.show()

def test_digit_accuracies(detector, testing_data, testing_labels):
    """
    Test the accuracy of the digit detection for each digit class.

    Args:
        detector (handWritingDetection): The handWritingDetection object.
        testing_data (numpy.ndarray): The testing dataset.
        testing_labels (numpy.ndarray): The labels for the testing dataset.

    Returns:
        dict: A dictionary containing the accuracy for each digit class.
    """
    # Initialize a dictionary to store the number of correct predictions for each digit
    correct_predictions_per_digit = {digit: 0 for digit in range(10)}

    # Initialize a dictionary to store the total predictions for each digit
    total_predictions_per_digit = {digit: 0 for digit in range(10)}

    for image, label in zip(testing_data, testing_labels):
        # Detect the digit
        predicted_digit = detector.detectWriting(image)
        total_predictions_per_digit[predicted_digit] += 1
        if predicted_digit == label:
            correct_predictions_per_digit[label] += 1

    # Calculate accuracy for each digit class
    digit_accuracies = {}
    for digit in range(10):
        accuracy = calculate_accuracy(correct_predictions_per_digit[digit], total_predictions_per_digit[digit])
        digit_accuracies[digit] = accuracy

    return digit_accuracies

def analyze_singular_values_per_digit(training_file, s_values):
    """
    Analyze the singular values of detectors trained with different numbers of basis vectors for each digit.

    Args:
    - training_file: Path to the training data file.
    - s_values: List of values for the number of basis vectors.

    Returns:
    - Dictionary containing singular values for each digit and each detector.
    """
    singular_values_per_digit_per_detector = {}

    # Iterate over each value of s
    for s in s_values:
        detector = train_model(training_file, k=400, s=s)
        singular_values= {}

        for digit in range(10):
            # Get the singular values for the current digit and detector
            _, singular_values[digit], _ = np.linalg.svd(detector.digit_matrices[digit])

        # Save the singular values for the current detector in the main dictionary
        singular_values_per_digit_per_detector[s] = singular_values

    return singular_values_per_digit_per_detector

def plot_singular_values_per_digit(singular_values_per_digit_per_detector):
    """
    Plot histograms of singular values for each digit and each detector using varying number of basis.

    Args:
    - singular_values_per_digit_per_detector: Dictionary containing singular values for each digit and each detector.

    Returns:
    - None
    """
    for s, singular_values_per_digit in singular_values_per_digit_per_detector.items():
        # Plot histograms of singular values for each digit
        for digit, singular_values in singular_values_per_digit.items():
            plt.hist(singular_values, bins=20, alpha=0.5, label=f'Digit {digit}')
        plt.xlabel('Singular Values')
        plt.ylabel('Frequency')
        plt.title(f'Singular Value Histogram for Detector with s={s}')
        plt.grid(True)
        plt.legend()
        plt.show()


def main():
    training_file = "handwriting_training_set.txt"
    testing_data_file = "handwriting_test_set.txt"
    testing_labels_file = "handwriting_test_set_labels.txt"
    testing_data = np.loadtxt(testing_data_file, delimiter=",")
    testing_labels = np.loadtxt(testing_labels_file)

    k = 400
    s_values = [5, 10, 15, 20]
    detectors = {}
    for s in s_values:
        detectors[s] = train_model(training_file, k, s)

    # Task i
    plot_accuracy_vs_basis_vectors(detectors, testing_data, testing_labels, s_values)

    # Task ii
    for s in s_values:
        digit_accuracies = test_digit_accuracies(detectors.get(s), testing_data, testing_labels)
        print(f"Accuracy for {s} basis: ")
        for digit, accuracy in digit_accuracies.items():
            print(f"Accuracy for digit {digit}: {accuracy} %")


    # Task iii
    singular_values_per_digit_per_detector = analyze_singular_values_per_digit(training_file, s_values)
    plot_singular_values_per_digit(singular_values_per_digit_per_detector)

if __name__ == "__main__":
    main()