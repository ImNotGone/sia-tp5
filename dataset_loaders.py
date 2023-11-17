from typing import List, Tuple

import numpy as np
from numpy._typing import NDArray


# Loads a 7x5 matrix from a file into a list of 7x5 matrices
# Each character in the file represents a value in the matrix
def load_numbers() -> List[NDArray]:
    arr = []

    rows = 7
    cols = 5
    file_name = "./data/ej3-digitos.txt"

    # read from file
    with open(file_name, "r") as file:
        # Group the files lines in groups of 7
        lines = file.readlines()

        # Strip \n from each line
        # Strip spaces from each line
        lines = [line.strip().replace(" ", "") for line in lines]

        line_groups = [lines[i : i + rows] for i in range(0, len(lines), rows)]

        # For each group of 7 lines
        for line_group in line_groups:
            # Create a 7x5 matrix
            matrix = np.zeros((rows, cols))

            # For each line
            for i, line in enumerate(line_group):
                # For each character in the line
                for j, char in enumerate(line):
                    # If the character is a 1, set the matrix value to 1
                    if char == "1":
                        matrix[i][j] = 1

            # Add the matrix to the list
            arr.append(matrix)

    return arr


def create_even_numbers_identifier_dataset() -> List[Tuple[NDArray, NDArray]]:
    numbers = load_numbers()

    # Create a list of tuples (input, expected_output)
    # The input is a 7x5 matrix that represents a number
    # The expected output are 2 neurons,
    # the first one is 1 if the number is even, 0 otherwise
    # the second one is 1 if the number is odd, 0 otherwise
    dataset = []

    for i, number in enumerate(numbers):
        # Flatten the matrix
        numbers[i] = number.flatten()

        # Create the expected output
        expected_output = np.zeros(2)
        expected_output[i % 2] = 1

        dataset.append((numbers[i], expected_output))

    return dataset


def create_numbers_identifier_dataset() -> List[Tuple[NDArray, NDArray]]:
    numbers = load_numbers()

    # Create a list of tuples (input, expected_output)
    # The input is a 7x5 matrix that represents a number
    # The expected output are 10 neurons,
    # Each neuron represents a number from 0 to 9
    # The expected output neuron is
    # 1 if the number is the same as the index of the neuron, 0 otherwise
    dataset = []

    for i, number in enumerate(numbers):
        # Flatten the matrix
        numbers[i] = number.flatten()

        # Create the expected output
        expected_output = np.zeros(10)
        expected_output[i] = 1

        dataset.append((numbers[i], expected_output))

    return dataset


def create_numbers_identifier_dataset_with_noise() -> List[Tuple[NDArray, NDArray]]:
    dataset = create_numbers_identifier_dataset()

    # Add noise to the dataset
    for i, (input, expected_output) in enumerate(dataset):
        # Add noise to the input
        noise = np.random.normal(0, 0.1, input.shape)
        dataset[i] = (input + noise, expected_output)

    return dataset


def create_xor_dataset() -> List[Tuple[NDArray, NDArray]]:
    input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    expected_output = np.array([[1], [1], [-1], [-1]])

    # Output is 1 0 for 1, 0 1 for -1
    expected_output = np.where(
        expected_output == 1, np.array([[1, 0]]), np.array([[0, 1]])
    )

    return [(input[i], expected_output[i]) for i in range(len(input))]

def create_xor_dataset_zero_one() -> List[Tuple[NDArray, NDArray]]:
    input = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    expected_output = np.array([[1], [1], [0], [0]])

    return [(input[i], expected_output[i]) for i in range(len(input))]

# ----- dataset generator -----


def get_dataset(config) -> List[Tuple[NDArray, NDArray]]:
    dataset_type = config["dataset"]

    if dataset_type == "even_numbers":
        return create_even_numbers_identifier_dataset()
    elif dataset_type == "numbers":
        return create_numbers_identifier_dataset()
    elif dataset_type == "numbers_with_noise":
        return create_numbers_identifier_dataset_with_noise()
    elif dataset_type == "xor":
        return create_xor_dataset()
    elif dataset_type == "xor_zero_one":
        return create_xor_dataset_zero_one()
    else:
        raise Exception("Dataset not found")
