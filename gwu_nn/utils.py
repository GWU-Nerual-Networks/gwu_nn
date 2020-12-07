from sklearn.utils import shuffle


def seperate_classes(x, y):
    """ This function puts all samples with the same label in the same list.
    Then, it puts them all in the same final list, accessible by their index.
    For example, sample images of digit "2" are found in x_results[2] and
    their corresponding labels (which are all 2s) are found in y_results[2].
    """
    x_results = []
    y_results = []
    for digit in range(10):
        y_digit = []
        x_digit = []
        for i in range(len(y)):
            if y[i] == digit:
                y_digit.append(y[i])
                x_digit.append(x[i])

        x_results.append(x_digit)
        y_results.append(y_digit)

    return x_results, y_results


def transofrm_to_binary(digit_1, digit_2, x, y):

    """
    This function allows the creation of X and y datasets for binary classification
    of digit_1 and digit_2. It obtains all digit_1 images and digit_2 images
    and returns x and y sets which consist of only the images of those two digits.
    """

    x_results, y_results = seperate_classes(x, y)

    x_digit_1 = x_results[digit_1]
    y_digit_1 = y_results[digit_1]

    x_digit_2 = x_results[digit_2]
    y_digit_2 = y_results[digit_2]


    x_bin = x_digit_1 + x_digit_2
    y_bin = y_digit_1 + y_digit_2

    # the label for the second digit becomes 1 while the label for the first digit it 0
    y_bin = [1 if y == digit_2 else 0 for y in y_bin]

    # shuffling the arrays in unision
    x_shuffled, y_shuffled = shuffle(x_bin, y_bin)

    return x_shuffled, y_shuffled
