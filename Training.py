import random as rand
from os.path import dirname
from os.path import join as pjoin

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from scipy.io import loadmat


def y2indicator(Y):
    """Converts output to 1X10 matrix where the 1 indicates the number from 0 to 9"""
    N = len(Y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, int(Y[i])] = 1
    return ind


def forward(X, w1, b1, w2, b2):
    """Forward generates the output layer Y and Hidden layer Z"""
    # input to layer 1
    Z = X.dot(w1)  # Z = 1 / (1 + np.exp(-(X.dot(w1) + b1)))
    # input to final layer with softmax applied
    A = Z.dot(w2)  # + b2

    # soft max
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z


# Derivatives for Weights and biases
def grad_w1(X, Z, T, Y, w2):
    """Derivative of weights of layer 1"""
    return X.T.dot(((Y - T).dot(w2.T) * (Z * (1 - Z))))


def derivative_b1(Z, T, Y, w2):
    """Derivative of biases of layer 1"""
    return ((Y - T).dot(w2.T) * (Z * (1 - Z))).sum(axis=0)


def grad_w2(Z, T, Y):
    """Derivative of weights of layer 2"""
    return Z.T.dot(Y - T)


def derivative_b2(T, Y):
    """Derivative of biases of layer 2"""
    return (Y - T).sum(axis=0)


def nest_list(list1, rows, columns):
    """Converts data from 1XN to RowsXColumns Data"""
    result = []
    start = 0
    end = columns
    for i in range(rows):
        result.append(list1[start:end])
        start += columns
        end += columns
    return result


def cost(T, Y):
    """The cost function for backpropogation of error"""
    return -(T * np.log(Y)).sum()


def data_split(data, labels, split):
    # split from 0 to 1
    data_length = split * len(data)
    training_data = data.tolist()
    training_labels = labels.tolist()
    test_data = []
    test_labels = []
    while len(training_data) > data_length:
        random = np.random.randint(0, len(training_data))
        test_data.append(training_data.pop(random))
        test_labels.append(training_labels.pop(random))

    return (
        np.array(training_data),
        np.array(training_labels),
        np.array(test_data),
        np.array(test_labels),
    )


def training(
    training_data, training_labels, M=50, epochs=50, evaluate=True, plot=False
):
    # Initialise training data
    X, T = training_data, y2indicator(training_labels)

    # Initialise Weights and Biases M is the number of hyperparameters
    D = 784
    K = 10
    w1 = np.random.randn(D, M)
    b1 = np.random.randn(1, M)
    w2 = np.random.randn(M, K)
    b2 = np.random.randn(1, K)
    learning_rate = 0.00001

    cost_plot = []
    # accuracy = []
    # balanced_accuracy = []
    for i in range(epochs):
        Y, Z = forward(X, w1, b1, w2, b2)
        w2 -= learning_rate * grad_w2(Z, T, Y)
        b1 -= learning_rate * derivative_b1(Z, T, Y, w2)
        w1 -= learning_rate * grad_w1(X, Z, T, Y, w2)
        b2 -= learning_rate * derivative_b2(T, Y)
        print(cost(T, Y))
        # if evaluate == True:
        #    accuracy.append

        cost_plot.append(cost(T, Y))

    if plot == True:
        plot.plot([n for n in range(1, epochs + 1)], cost_plot)
        plot.show()

    return (w1, b1, w2, b2)
    # if evaluate == True:
    #    return (w1, b1, w2, b2)
    # else:
    #   return (w1, b1, w2, b2)


def model_test(w1, b1, w2, b2, data_point):
    # per single data point
    result_layer_1 = 1 / (1 + np.exp(-(data_point.dot(w1) + b1)))
    result_layer_2 = np.exp(result_layer_1.dot(w2) + b2)
    result_layer_3 = result_layer_2 / result_layer_2.sum()
    return result_layer_3


def train(training_data, training_labels):
    M, epochs = 128, 50
    w1, b1, w2, b2 = training(training_data, training_labels, M, epochs)

    # np.savetxt("weights and biases\w1.csv", w1, delimiter=",")
    # np.savetxt("weights and biases\\b1.csv", b1, delimiter=",")
    # np.savetxt("weights and biases\w2.csv", w2, delimiter=",")
    # np.savetxt("weights and biases\\b2.csv", b2, delimiter=",")


def load_model():
    w1 = np.loadtxt("weights and biases\M=128\w1.csv", delimiter=",")
    b1 = np.loadtxt("weights and biases\M=128\\b1.csv", delimiter=",")
    w2 = np.loadtxt("weights and biases\M=128\w2.csv", delimiter=",")
    b2 = np.loadtxt("weights and biases\M=128\\b2.csv", delimiter=",")
    return (w1, b1, w2, b2)


def gen_confusion_matrix(w1, b1, w2, b2, test_data, test_labels):
    confusion_matrix = np.zeros((10, 10))
    for i, data in enumerate(test_data):
        result = np.argmax(model_test(w1, b1, w2, b2, data))
        confusion_matrix[int(test_labels[i])][result] += 1
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix):
    fig, ax = plot.subplots(figsize=(7.5, 7.5))
    ax.matshow(confusion_matrix, cmap=plot.cm.Blues, alpha=0.7)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=int(confusion_matrix[i, j]),
                va="center",
                ha="center",
                size="xx-large",
            )
    plot.show()


def main():
    # Data import
    mnist = loadmat(r"input\mnist-original.mat")
    # convert training data to normalised vector
    mnist_data = (mnist["data"].T) / 256
    mnist_label = mnist["label"][0]

    # split into the training and test set
    training_data, training_labels, test_data, test_labels = data_split(
        mnist_data, mnist_label, 0.7
    )
    k = rand.randint(0, len(test_data))
    np.savetxt("weights and biases\example.csv", test_data[k], delimiter=",")

    train = True

    if train:
        train(training_data, training_labels)
    else:
        w1, b1, w2, b2 = load_model()

    # generate confusion matrix for the models predictions of the test set
    # confusion_matrix = gen_confusion_matrix(w1, b1, w2, b2, test_data, test_labels)
    # plot_confusion_matrix(confusion_matrix)

    j = np.random.randint(1, 2000)
    # print(model_test(w1, b1, w2, b2, test_data[j]))
    plot.imshow(nest_list(test_data[j], 28, 28))
    print(test_labels[j])
    plot.show()


if __name__ == "__main__":
    main()
