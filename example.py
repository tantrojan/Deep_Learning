# Introductory Example

import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist # 28*28 images of hand-written digits 0-9

'''
The MNIST database (Modified National Institute of Standards and Technology database) is a 
large database of handwritten digits that is commonly used for training various image processing systems.
It was created by "re-mixing" the samples from NIST's original datasets.
'''

(x_train, y_train), (x_test, y_test)  = mnist.load_data()

# normalize the dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Visualize the dataset
'''
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()
'''

# Creating the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) #Input Layer, Flatten is used to convert the n*m shape into a single layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #Hidden Layer, Parameters = (number_of_neurons=128, activation_function = ReLu (Rectified Linear))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #Hidden Layer, Parameters = (number_of_neurons=128, activation_function = ReLu (Rectified Linear))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(x_train, y_train, epochs=3)


# Calculate the validation loss
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# Use the model to predict values
predictions = model.predict([x_test])

predictions = np.argmax(predictions)