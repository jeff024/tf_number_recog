# importing related library
import tensorflow as tf
import numpy as np

print(tf.__version__)

# tensorflow.keras
print(dir(tf.keras.datasets))

mnist = tf.keras.datasets.mnist

# load dataset to ~/.keras\datasets，Example：C:\Users\dennis\.keras\datasets\mnist.npz，around 11 MB
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print (x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print (type(x_train))

x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
print (x_train.shape, x_test.shape)

# display one data in the dataset
import matplotlib.pyplot as plt
image_index = 59999
print("The label is ", y_train[image_index])
# plt.imshow(x_train[image_index], cmap='Greys')
plt.imshow(x_train[image_index])
plt.show()

# transforming dataset
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

# TensorFlow dataset formats：[n, h, w, c]，which is number, heigth, width, channel
x_train = x_train.reshape(x_train.shape[0], 32, 32, 1) 
x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
print (x_train.shape, x_test.shape)

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

"Part 2: building the model"

# define the model
# key elements in this model： tf.keras.Model 和 tf.keras.layers
# loss function： tf.keras.losses
# optimizer： tf.keras.optimizer
# evaluation： tf.keras.metrics

"method 1: class"
# class LeNet(tf.keras.Model):
#     def __init__(self):
#         super().__init__()     
#         self.conv_layer_1 = tf.keras.layers.Conv2D(
#                 filters=6,
#                 kernel_size=(5, 5),
#                 padding='valid',
#                 activation=tf.nn.relu)
#         self.pool_layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')

#         self.conv_layer_2 = tf.keras.layers.Conv2D(
#                 filters=16,
#                 kernel_size=(5, 5),
#                 padding='valid',
#                 activation=tf.nn.relu)
#         self.pool_layer_2 = tf.keras.layers.MaxPooling2D(padding='same')

#         self.flatten = tf.keras.layers.Flatten()

#         self.fc_layer_1 = tf.keras.layers.Dense(
#                 units=120,
#                 activation=tf.nn.relu)
#         self.fc_layer_2 = tf.keras.layers.Dense(
#                 units=84,
#                 activation=tf.nn.relu)
#         self.output_layer = tf.keras.layers.Dense(
#                 units=10,
#                 activation=tf.nn.softmax)

#     def call(self, inputs):   # [batch_size, 28, 28, 1]
#         x = self.conv_layer_1(inputs)
#         x = self.pool_layer_1(x)
#         x = self.conv_layer_2(x)
#         x = self.pool_layer_2(x)
#         x = self.flatten(x)
#         x = self.fc_layer_1(x)
#         x = self.fc_layer_2(x)
#         output = self.output_layer(x)

#         return output

# model = LeNet()
"method 2: sequence"

model = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu, input_shape=(32,32,1)),#relu
 tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same'),
 tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu),
 tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same'),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(units=120, activation=tf.nn.relu),
#  tf.keras.layers.Conv2D(filters=120, kernel_size=(5,5),strides=(1,1),activation='tanh',padding='valid'),
#  tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(units=84, activation=tf.nn.relu),
 tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
 ])



# displaying model
model.summary()

"part 3: train the model"


num_epochs = 10
batch_size = 64
learning_rate = 0.001

# optimizer
adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

model.compile(optimizer=adam_optimizer,
       loss=tf.keras.losses.sparse_categorical_crossentropy,
       metrics=['accuracy'])

import datetime
start_time = datetime.datetime.now()

model.fit(x=x_train,
     y=y_train,
     batch_size=batch_size,
     epochs=num_epochs)
end_time = datetime.datetime.now()
time_cost = end_time - start_time
print ("time_cost = ", time_cost)

# saving the model
#from google.colab import drive
#drive.mount('/gdrive')

model.save('lenet_model.h5')

# evaluation
print(model.evaluate(x_test, y_test))  # loss value & metrics values

# prediction
image_index = 4444
print (x_test[image_index].shape)
plt.imshow(x_test[image_index].reshape(32, 32),cmap='Greys')

pred = model.predict(x_test[image_index].reshape(1, 32, 32, 1))
print(pred.argmax())  # argmax returns the max number of indexhttps://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.argmax.html