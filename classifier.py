from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# Load the mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Some info on the train and test sets
print("Train images shape: {0}".format(train_images.shape))
print("Number of train labels: {0}".format(len(train_labels)))
print("Train labels preview: {0}".format(train_labels[:10]))

print("Test images shape: {0}".format(test_images.shape))
print("Number of test labels: {0}".format(len(test_labels)))
print("Test labels preview: {0}".format(test_labels[:10]))

# Create the model
network = models.Sequential()

# Set up the architecture - Add the layers - The last one is of size 10 as we are
# classifying against 10 possible digits.
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

# Compile the model
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Prepare the image data
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

# Prepare the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Fit the model
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluate model on the test data
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("Test loss: {0}".format(test_loss))
print("Test accuracy: {0}".format(test_acc))
