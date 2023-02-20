import matplotlib.pyplot as plot
import numpy as np

from scipy.io import loadmat


class Linear:
    """A single linear layer with a logistic activation function,  with weights and biases"""

    def __init__(
        self,
        layer_length,
        prev_layer_length,
        next_layer_length,
    ):

        self.L = layer_length
        self.D = prev_layer_length
        self.O = next_layer_length

        self.weights_1 = np.random.randn(self.D, self.L)  # D by L matrix
        self.biases_1 = np.random.randn(self.L)  # 1 by L matrix

        self.weights_2 = np.random.randn(self.L, self.O)  # L by O matrix
        self.biases_2 = np.random.randn(self.O)  # 1 by O matrix

        self.cost_array = []

    def forward(self, inputs):
        """Moves forward in the calculation returning hidden and output layer values"""

        x = inputs.dot(self.weights_1) + self.biases_1
        x = 1 / (1 + np.exp(-x))

        hidden = x.copy()

        x = x.dot(self.weights_2) + self.biases_2
        x = 1 / (1 + np.exp(-x))

        output = self.softmax(x)
        return hidden, output

    def train(self, epochs, train_inputs, train_labels, train_rate):
        """Changes the values of the weights and biases using gradient descent during training"""
        # Derivatives for Weights and biases
        for i in range(epochs):
            hidden, outputs = self.forward(train_inputs)

            grad_w1 = train_inputs.T.dot(
                (outputs - train_labels).dot((self.weights_2).T)
                * (hidden * (1 - hidden))
            )

            grad_b1 = (
                (outputs - train_labels).dot((self.weights_2).T)
                * (hidden * (1 - hidden))
            ).sum(axis=0)

            grad_w2 = hidden.T.dot(outputs - train_labels)

            grad_b2 = (outputs - train_labels).sum(axis=0)

            self.weights_1 -= train_rate * grad_w1
            self.biases_1 -= train_rate * grad_b1
            self.weights_2 -= train_rate * grad_w2
            self.biases_2 -= train_rate * grad_b2

            cost = sum(sum(self.cost(train_labels, outputs)))
            if i % 100 == 0:
                print(f"At epoch:{i}, the loss was {cost}")

    def load_model(self, folder_path: str):
        """Load in the weights and biases from the saved model"""
        self.weights_1 = np.loadtxt(f"{folder_path}\w1.csv", delimiter=",")
        self.biases_1 = np.loadtxt(f"{folder_path}\\b1.csv", delimiter=",")
        self.weights_2 = np.loadtxt(f"{folder_path}\w2.csv", delimiter=",")
        self.biases_2 = np.loadtxt(f"{folder_path}\\b2.csv", delimiter=",")

    def model_test(self, input):
        """test a single input"""
        result_layer_1 = 1 / (1 + np.exp(-(input.dot(self.weights_1) + self.biases_1)))
        result_layer_2 = np.exp(result_layer_1.dot(self.weights_2) + self.biases_2)
        result_layer_3 = result_layer_2 / result_layer_2.sum()
        return result_layer_3

    def softmax(self, input):
        """performs the softmax operation on the input numpy array"""
        exp_input = np.exp(input)
        return exp_input / exp_input.sum(axis=1, keepdims=True)

    def cost(self, labels, outputs):
        """Cost function for the error of the model"""
        return -(labels * np.log(outputs))


def y2indicator(training_labels):
    """Converts output to 1X10 matrix where the 1 indicates the number from 0 to 9"""
    N = len(training_labels)

    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, int(training_labels[i])] = 1
    return ind


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


def gen_confusion_matrix(model, test_data, test_labels):
    confusion_matrix = np.zeros((10, 10))
    for i, data in enumerate(test_data):
        result = np.argmax(model.model_test(data))
        confusion_matrix[int(test_labels[i])][result] += 1
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix):
    # Set up figure and axes
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


def main():
    # Data import
    mnist = loadmat(r"input\mnist-original.mat")
    # Convert training data to normalised vector
    mnist_data = (mnist["data"].T) / 256
    mnist_label = mnist["label"][0]

    # Hyperparameters
    input_length = 784
    hidden_layer_length = 100
    output_length = 10
    train_rate = 1e-5
    epochs = 1000

    # Split into the training and test set
    training_data, training_labels, test_data, test_labels = data_split(
        mnist_data, mnist_label, 0.7
    )

    # Initialise the training labels
    training_labels = y2indicator(training_labels)

    # Create the model of one Linear layer mappping input to output
    model = Linear(hidden_layer_length, input_length, output_length)

    train_model = True
    save_model = True
    # Train the model
    if train_model:
        model.train(
            epochs,
            training_data,
            training_labels,
            train_rate,
        )
        if save_model:
            np.savetxt(
                f"weights and biases/M={hidden_layer_length}/w1.csv",
                model.weights_1,
                delimiter=",",
            )
            np.savetxt(
                f"weights and biases/M={hidden_layer_length}/b1.csv",
                model.biases_1,
                delimiter=",",
            )
            np.savetxt(
                f"weights and biases/M={hidden_layer_length}/w2.csv",
                model.weights_2,
                delimiter=",",
            )
            np.savetxt(
                f"weights and biases/M={hidden_layer_length}/b2.csv",
                model.biases_2,
                delimiter=",",
            )
    else:
        model.load_model("weights and biases/M=50")

    # k = rand.randint(0, len(test_data))
    # np.savetxt("weights and biases\example.csv", test_data[k], delimiter=",")

    # generate confusion matrix for the models predictions of the test set
    confusion_matrix = gen_confusion_matrix(model, test_data, test_labels)
    plot_confusion_matrix(confusion_matrix)

    j = np.random.randint(1, 2000)
    print(model.model_test(test_data[j]))
    plot.imshow(nest_list(test_data[j], 28, 28))
    print(test_labels[j])
    plot.show()


if __name__ == "__main__":
    main()
