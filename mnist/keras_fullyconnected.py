from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
import seaborn as sns

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10

# Normalize data [0,255] -> [0,1]
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

# Apply one-hot encoding to the labels
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Break training data into training and validation sets
(x_train, x_valid) = x_train[:50000], x_train[50000:]
(y_train, y_valid) = y_train[:50000], y_train[50000:]

# Model definition
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
#model.add(Flatten(input_shape=x_train.shape[1:]))

# First layer
model.add(Dense(100, use_bias=False, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Second layer
model.add(Dense(100, use_bias=False, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Third layer
model.add(Dense(100, use_bias=False, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Fourth layer
model.add(Dense(100, use_bias=False, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Fifth layer
model.add(Dense(100, use_bias=False, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output 
model.add(Dense(num_classes, use_bias=False, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

print(model.summary())

# Set hyperparameters
batch_size = 128
epochs = 50
learning_rate = 0.1

optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
#optimizer = Adam()

model.compile(optimizer=optimizer, loss='categorical_crossentropy',
	metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
	validation_data=(x_valid, y_valid), verbose=1, shuffle=True)

score = model.evaluate(x_test, y_test, batch_size=128)

print(f'\nTest accuracy: {score[1]:>5.3f}')


figure = plt.figure()
sns.set_context('talk')
plt.subplot(1,2,1)
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.xticks(range(1, epochs+1))
plt.ylabel('Accuracy')
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.xticks(range(1, epochs+1))
plt.ylabel('Loss')
plt.legend()

plt.show()

