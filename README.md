# NeuralNetwork-Implementations

Code containing implementations of neural networks for different learning tasks.
As of now, models have been implemented in Tensorflow using it's core API. 

## Structure:
### **Iris:**
* **iris_network.py** - Pure Python/Numpy implementation of a two layer neural network. The network is a binary classifier that tries to learn features for a single target variable in the Iris dataset.
* **iris_tensorflow.py** - Neural Network model built using Tensorflow. The network uses Softmax activation for multiclass prediction on the Iris dataset.

--------

### **Mnist:**
* **tensorflow_mnist.py** - Command line interface to experiment with hyperparameters of deep learning models. The MNIST dataset of handwritten characters has been used.

```
usage: tensorflow_mnist.py [-h] [-e EPOCHS] [-r LEARNRATE] [-l LAYERS]
                           [-b BATCHSIZE]

Deep network for classifying MNIST images.

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of Epochs for training the network.
  -r LEARNRATE, --learnrate LEARNRATE
                        Learning Rate used in training.
  -l LAYERS, --layers LAYERS
                        Comma-delimited list of the number of nodes in the
                        hidden layers.
  -b BATCHSIZE, --batchsize BATCHSIZE
                        Total batch size used in training.


#Example

$ python3 tensorflow_mnist.py -e 100 -r 0.01 -b 256 -l "20, 10, 20"

                Running session with
                    Epochs: 100
                    Learning Rate: 0.01000
                    Batch Size: 256
                    Num of hidden layers (nodes): 3 [20, 10, 20]

    Epoch:     1 Loss:     2.2956 Validation Accuracy: 0.1124
    Epoch:     2 Loss:     2.3031 Validation Accuracy: 0.1126
    ...
    Epoch:    99 Loss:     0.0949 Validation Accuracy: 0.9330
    Epoch:   100 Loss:     0.1376 Validation Accuracy: 0.9394
    Testing Accuracy: 0.935
```


