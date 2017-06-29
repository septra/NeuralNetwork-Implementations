## Tensorflow for MNIST

This is a command line interface for experimenting with hyper-parameters in deep learning models implemented in Tensorflow.

The MNIST dataset is hosted on Yann LeCun's site [here](http://yann.lecun.com/exdb/mnist/).
Tensorflow takes care of downloading, extracting and formatting the data into the proper form for use in our neural networks.

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
```

Deeper and wider networks don't necessarily result in better performance as shown by the following experiments.


### Summary:
| Epochs    | Learning Rate    | Number of Hidden Layers    | Hidden Node Architecture    | Test Accuracy |
| --------- |-----------------:| --------------------------:|:---------------------------:| :-------------|
| 10        | 0.01             | 1                          | [20]                        | 0.928
| 30        | 0.01             | 1                          | [20]                        | 0.948
| 30        | 0.01             | 3                          | [20, 10, 20]                | 0.909
| 100       | 0.01             | 3                          | [20, 10, 20]                | 0.935


```
$ python3 tensorflow_mnist.py -e 10 -r 0.01 -b 256 -l "20"
    
            Running session with
            	Epochs: 10
                Learning Rate: 0.01000
                Batch Size: 256
                Num of hidden layers (nodes): 1 [20]

Epoch:     1 Loss:     1.3147 Validation Accuracy: 0.6026
Epoch:     2 Loss:     0.7626 Validation Accuracy: 0.8064
...
Epoch:     9 Loss:     0.2120 Validation Accuracy: 0.9312
Epoch:    10 Loss:     0.2111 Validation Accuracy: 0.9324
Testing Accuracy: 0.928
```

```
$ python3 tensorflow_mnist.py -e 30 -r 0.01 -b 256 -l "20"

            Running session with
            	Epochs: 30
            	Learning Rate: 0.01000
            	Batch Size: 256
            	Num of hidden layers (nodes): 1 [20]

Epoch:     1 Loss:     1.0352 Validation Accuracy: 0.6954
Epoch:     2 Loss:     0.6237 Validation Accuracy: 0.8196
...
Epoch:    29 Loss:     0.0523 Validation Accuracy: 0.9512
Epoch:    30 Loss:     0.1105 Validation Accuracy: 0.9514
Testing Accuracy: 0.948
```

```
$ python3 tensorflow_mnist.py -e 30 -r 0.01 -b 256 -l "20, 10, 20"

            Running session with
            	Epochs: 30
            	Learning Rate: 0.01000
            	Batch Size: 256
            	Num of hidden layers (nodes): 3 [20, 10, 20]

Epoch:     1 Loss:     2.3249 Validation Accuracy: 0.1140
Epoch:     2 Loss:     2.2874 Validation Accuracy: 0.1144
...
Epoch:    29 Loss:     0.2805 Validation Accuracy: 0.9136
Epoch:    30 Loss:     0.2973 Validation Accuracy: 0.9168
Testing Accuracy: 0.909
```

```
python3 tensorflow_mnist.py -e 100 -r 0.01 -b 256 -l "20, 10, 20"

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
