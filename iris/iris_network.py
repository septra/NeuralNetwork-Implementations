from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np

data = load_iris()

X = data.data
y = data.target

# Predict only for versicolor
y_proc = np.array([1 if x==1 else 0 for x in y])

X_train, X_test, y_train, y_test = train_test_split(X, y_proc, test_size=0.2, random_state=42)


# no of input features = 4
# no of hidden nodes = 5
# no of output nodes = 1

weights = {
           "input_hidden": np.random.normal(0, 1, (4, 5)),
           "hidden_output": np.random.normal(0, 1, (5, 1))
      }

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Keeping the token value of the input (z) in the function signature.
def sigmoid_der(z, x):
    return x * (1 - x)

def predict(inp):
    hidden_layer = sigmoid(np.dot(inp, weights['input_hidden']))
    output_layer = sigmoid(np.dot(hidden_layer, weights['hidden_output']))
    return np.array([1 if x >= 0.5 else 0 for x in output_layer])

training_accuracy = []
test_accuracy = []
error_graph = []

learning_rate = 0.008

def train(epochs):
    for epoch in range(epochs):
        in_layer = X_train     # (120, 4)

        z_hidden = np.dot(in_layer, weights['input_hidden'])  # (120, 4) * (4, 5) = (120, 5)
        hidden_layer = sigmoid(z_hidden)   # (120, 5)

        z_output = np.dot(hidden_layer, weights['hidden_output'])  # (120, 5) * (5, 1) = (120, 1)
        output_layer = sigmoid(z_output) # (120, 1)

        output_error = y_train[:, None] - output_layer # (120, 1)
        output_error_term = output_error * sigmoid_der(z_output, output_layer) # (120, 1) * (120, 1) = (120, 1)

        binary_output = (output_layer >= 0.5).astype(int)

        if (epoch % 1000 == 0):
            print("Error: " + str(np.mean(np.abs(output_error))) + " ; Accuracy: " + str(accuracy_score(y_train, binary_output)))

        training_accuracy.append(accuracy_score(y_train, binary_output))
        test_accuracy.append(accuracy_score(y_test, predict(X_test)))
        error_graph.append(np.mean(np.abs(output_error)))
           

        hidden_error = np.dot(output_error_term, weights['hidden_output'].T) # (120, 1) * (1, 5) = (120, 5)
        hidden_error_term = hidden_error * sigmoid_der(z_hidden, hidden_layer)  # (120, 5) * (120, 5) = (120, 5)

        weights['input_hidden'] += learning_rate * np.dot(in_layer.T, hidden_error_term)  # (4, 120) * (120, 5) = (4,5)
        weights['hidden_output'] += learning_rate * np.dot(hidden_layer.T, output_error_term) # (5, 120) * (120, 1) = (5, 1)

def plot():
    plt.plot(training_accuracy, 'blue', label = 'Training Accuracy')
    plt.plot(test_accuracy, 'green', label = 'Test Accuracy')
    plt.plot(error_graph, 'red', label = 'Loss')
    plt.legend()
    plt.show()


print(accuracy_score(y_test, predict(X_test)))

train(30000)
plot()

