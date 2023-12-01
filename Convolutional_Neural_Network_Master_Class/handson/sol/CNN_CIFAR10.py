# Build a CNN model on CIFAR-10 dataset by applying few regularization techniques like drop out and data augmentation.
import pandas as pd
from matplotlib import pyplot
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CIFAR10 dataset
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i])
pyplot.show()

#one hot encoding of the label values
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# convert the data type to 'int' and also normalize the image pixel
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


#Design the network
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
#model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))
#model.summary()


opt = SGD(lr = 0.001, momentum = 0.9)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 5, batch_size = 64)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)

print('> %.3f' % (test_acc * 100.0))


# Data Augmentation

#generate the data 
datagen = ImageDataGenerator(horizontal_flip = True)

it_train = datagen.flow(x_train, y_train, batch_size = 64)

steps = int(x_train.shape[0] / 64)
history = model.fit_generator(it_train, steps_per_epoch = steps, epochs = 5, validation_data = (x_test, y_test), verbose = 0)

test_loss1, test_acc1 = model.evaluate(x_test, y_test, verbose = 0)

test_acc1


###########
#Design the network
model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
model1.add(layers.MaxPooling2D((2,2)))

model1.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model1.add(layers.MaxPooling2D((2,2)))

model1.add(layers.Conv2D(64, (3,3), activation = 'relu'))
#model.summary()

model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))

model1.add(layers.Dense(10, activation='softmax'))
#model.summary()



model1.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])

model1.fit(x_train, y_train, epochs = 5, batch_size = 64)
test_loss2, test_acc2 = model1.evaluate(x_test, y_test, verbose = 2)

print('> %.3f' % (test_acc2 * 100.0))






























