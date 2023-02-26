import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import loadmat


class Linear:
    """A single linear layer model with a logistic activation function,  with weights and biases"""

    def __init__(
        self,
        layer_length: int,
        prev_layer_length: int,
        next_layer_length: int,
    ):

        self.L = layer_length
        self.D = prev_layer_length
        self.O = next_layer_length

        self.weights_1 = np.random.randn(self.D, self.L)  # D by L matrix
        self.biases_1 = np.random.randn(self.L)  # 1 by L matrix

        self.weights_2 = np.random.randn(self.L, self.O)  # L by O matrix
        self.biases_2 = np.random.randn(self.O)  # 1 by O matrix

        self.cost_array = []

    def load_model(self, folder_path: str):
        """Load in the weights and biases from the saved model"""
        self.weights_1 = np.loadtxt(f"{folder_path}\w1.csv", delimiter=",")
        self.biases_1 = np.loadtxt(f"{folder_path}\\b1.csv", delimiter=",")
        self.weights_2 = np.loadtxt(f"{folder_path}\w2.csv", delimiter=",")
        self.biases_2 = np.loadtxt(f"{folder_path}\\b2.csv", delimiter=",")

    def import_data(self, training_data, training_labels):
        """Import the training data as numpy arrays"""
        self.training_data = training_data
        self.training_labels = training_labels

    def forward(self, inputs):
        """Propogates values through the model returning hidden and output layer values"""

        x = inputs.dot(self.weights_1) + self.biases_1
        x = 1 / (1 + np.exp(-x))

        hidden = x.copy()

        x = x.dot(self.weights_2) + self.biases_2

        output = self.softmax(x)
        return hidden, output

    def get_training_batch(self, batch_size: int, training_data, training_labels):
        """Get a random training batch based on the batch size"""
        data_length = batch_size
        training_data = training_data.tolist()
        training_labels = training_labels.tolist()

        while len(training_data) > data_length:
            random = np.random.randint(0, len(training_data))
            training_data.pop(random)
            training_labels.pop(random)

        return (
            np.array(training_data),
            np.array(training_labels),
        )

    def train(self, epochs: int, batch_size, train_rate, evaluation_interval: int):
        """Changes the values of the weights and biases using gradient descent during training"""
        # Derivatives for Weights and biases
        print(
            f"Training with hidden layer of length {self.L}, for {epochs} epochs at a train rate of {train_rate} "
        )

        train_inputs, train_labels = self.get_training_batch(
            batch_size, self.training_data, self.training_labels
        )

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

            if i % evaluation_interval == 0:
                print(f"At epoch:{i}, the loss was {cost}")

    def model_test(self, input):
        """test a single input"""
        result_layer_1 = 1 / (1 + np.exp(-(input.dot(self.weights_1) + self.biases_1)))
        result_layer_2 = result_layer_1.dot(self.weights_2) + self.biases_2
        result_layer_3 = np.exp(result_layer_2) / sum(np.exp(result_layer_2))
        return result_layer_3

    def softmax(self, input):
        """performs the softmax operation on the input numpy array"""
        exp_input = np.exp(input)
        return exp_input / exp_input.sum(axis=1, keepdims=True)

    def cost(self, labels, outputs):
        """Cost function for the error of the model"""
        return -(labels * np.log(outputs))

    def save(self, path):
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        np.savetxt(f"{path}/w1.csv", self.weights_1, delimiter=",")
        np.savetxt(f"{path}/b1.csv", self.biases_1, delimiter=",")
        np.savetxt(f"{path}/w2.csv", self.weights_2, delimiter=",")
        np.savetxt(f"{path}/b2.csv", self.biases_2, delimiter=",")


def y2indicator(training_labels):
    """Converts output to 1X10 matrix where the 1 indicates the number from 0 to 9"""
    N = len(training_labels)

    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, int(training_labels[i])] = 1
    return ind


def data_split(data, labels, split):
    """split the data and labels by a fraction betwee 0 to 1"""
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


def plot_confusion_matrix(model, test_data, test_labels):
    """Returns a matplotlib ax object of the confusion matrix over the data and labels provided"""

    confusion_matrix = np.zeros((10, 10))
    for i, data in enumerate(test_data):

        result = np.argmax(model.model_test(data))
        confusion_matrix[np.argmax(test_labels[i])][result] += 1

    # Set up figure and axes
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.7)
    ax.set_xlabel("Actual")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Predicted")

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
    return ax


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


def test(test_data, test_labels, model, j):
    """Test the model at a single data point"""

    model_prediction = np.argmax(model.model_test(test_data[j]))
    label = test_labels[j]

    fig, ax = plt.subplots()
    plt.imshow(nest_list(test_data[j], 28, 28))
    print(f"test label: {label}, model prediction: {model_prediction}")
    return ax


def main():

    # Hyper parameters
    input_length = 784
    output_length = 10
    hidden_layer_length = 150
    train_rate = 1e-4
    batch_size = 7000
    epochs = 1000
    evaluation_interval = 100

    # Data import
    mnist = loadmat(r"input\mnist-original.mat")
    # Convert training data to normalised vector
    data = (mnist["data"].T) / 256
    print(f"loaded {len(data)} training samples")
    labels = y2indicator(mnist["label"][0])

    # create a linear model
    model = Linear(hidden_layer_length, input_length, output_length)

    # Should the resulting model weights and biases be saved
    save_model = True

    # Train the model
    path = f"models/epochs={epochs},M={hidden_layer_length}"

    # Separate into a training and test set

    training_data, training_labels, test_data, test_labels = data_split(
        data, labels, 0.9
    )

    # load the data into the model for training
    model.import_data(training_data, training_labels)

    # retrain the model if a model exists with the same hyperparameters
    retrain = True

    if retrain == False:
        if os.path.exists(path):
            model.load_model(path)
        else:
            model.train(epochs, batch_size, train_rate, evaluation_interval)
    else:
        model.train(epochs, batch_size, train_rate, evaluation_interval)

    if save_model:
        if os.path.exists(path):
            shutil.rmtree(path)
        model.save(path)

    # generate confusion matrix for the models predictions of the test set
    plot_confusion_matrix(model, test_data, test_labels)
    path = "figures"
    try:
        os.mkdir(path)
    except OSError as error:
        pass

    figure_path = f"figures/epochs={epochs},M={hidden_layer_length}.png"

    if os.path.exists(figure_path):
        os.remove(figure_path)

    plt.savefig(figure_path)
    plt.show()


if __name__ == "__main__":
    main()
