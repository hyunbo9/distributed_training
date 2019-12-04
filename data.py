import numpy as np
import tensorflow as tf


(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

train_x = train_x.reshape((60000, 28, 28, 1))
train_x = np.asarray(train_x, dtype=np.float32) / 255
train_y = np.eye(10)[train_y]

test_x = test_x.reshape((10000,28,28,1))
test_x = np.asarray(test_x, dtype=np.float32) / 255
test_y = np.eye(10)[test_y]
print(train_y.shape)


epochs = 77
batch_size = 50
ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
ds = ds.repeat(epochs).batch(batch_size)