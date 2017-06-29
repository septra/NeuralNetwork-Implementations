## Tensorflow for MNIST

This is a command line interface for experimenting with different hyper-parameters in deep learning model implemented in Tensorflow.

The MNIST dataset is hosted on Yann LeCun's site [here](http://yann.lecun.com/exdb/mnist/).
Tensorflow takes care of downloading, extracting and formatting the data into the proper form foruse in our neural network.

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

```
python3 tensorflow_mnist.py -e 10 -r 0.01 -b 256 -l "20"
    
            Running session with
            	Epochs: 10
            Learning Rate: 0.01000
            Batch Size: 256
            Num of hidden layers (nodes): 1 [20]
Epoch:     1 Loss:     1.3147 Validation Accuracy: 0.6026
Epoch:     2 Loss:     0.7626 Validation Accuracy: 0.8064
Epoch:     3 Loss:     0.5697 Validation Accuracy: 0.8708
Epoch:     4 Loss:     0.5114 Validation Accuracy: 0.9012
Epoch:     5 Loss:     0.3414 Validation Accuracy: 0.9088
Epoch:     6 Loss:     0.3370 Validation Accuracy: 0.9188
Epoch:     7 Loss:     0.2324 Validation Accuracy: 0.9224
Epoch:     8 Loss:     0.2400 Validation Accuracy: 0.9260
Epoch:     9 Loss:     0.2120 Validation Accuracy: 0.9312
Epoch:    10 Loss:     0.2111 Validation Accuracy: 0.9324
Testing Accuracy: 0.928
```

```
python3 tensorflow_mnist.py -e 30 -r 0.01 -b 256 -l "20, 10, 20"


            Running session with
            	Epochs: 30
            	Learning Rate: 0.01000
            	Batch Size: 256
            	Num of hidden layers (nodes): 3 [20, 10, 20]

Epoch:     1 Loss:     2.3249 Validation Accuracy: 0.1140
Epoch:     2 Loss:     2.2874 Validation Accuracy: 0.1144
Epoch:     3 Loss:     2.3014 Validation Accuracy: 0.1156
Epoch:     4 Loss:     2.3122 Validation Accuracy: 0.1352
Epoch:     5 Loss:     2.1918 Validation Accuracy: 0.1576
Epoch:     6 Loss:     1.7434 Validation Accuracy: 0.3376
Epoch:     7 Loss:     1.5235 Validation Accuracy: 0.4198
Epoch:     8 Loss:     1.3055 Validation Accuracy: 0.4890
Epoch:     9 Loss:     1.2293 Validation Accuracy: 0.5536
Epoch:    10 Loss:     1.0090 Validation Accuracy: 0.6352
Epoch:    11 Loss:     0.9728 Validation Accuracy: 0.6814
Epoch:    12 Loss:     0.9119 Validation Accuracy: 0.7156
Epoch:    13 Loss:     0.8846 Validation Accuracy: 0.7406
Epoch:    14 Loss:     0.7544 Validation Accuracy: 0.7820
Epoch:    15 Loss:     0.5997 Validation Accuracy: 0.8012
Epoch:    16 Loss:     0.6090 Validation Accuracy: 0.8176
Epoch:    17 Loss:     0.5521 Validation Accuracy: 0.8356
Epoch:    18 Loss:     0.5356 Validation Accuracy: 0.8518
Epoch:    19 Loss:     0.6342 Validation Accuracy: 0.8614
Epoch:    20 Loss:     0.4190 Validation Accuracy: 0.8764
Epoch:    21 Loss:     0.3555 Validation Accuracy: 0.8830
Epoch:    22 Loss:     0.4400 Validation Accuracy: 0.8976
Epoch:    23 Loss:     0.2856 Validation Accuracy: 0.8944
Epoch:    24 Loss:     0.3515 Validation Accuracy: 0.8990
Epoch:    25 Loss:     0.3619 Validation Accuracy: 0.8968
Epoch:    26 Loss:     0.3141 Validation Accuracy: 0.9032
Epoch:    27 Loss:     0.2674 Validation Accuracy: 0.9102
Epoch:    28 Loss:     0.2471 Validation Accuracy: 0.9094
Epoch:    29 Loss:     0.2805 Validation Accuracy: 0.9136
Epoch:    30 Loss:     0.2973 Validation Accuracy: 0.9168
Testing Accuracy: 0.909
```
