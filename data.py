import numpy as np
import tensorflow as tf

#(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
#
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# keras = tf.keras
# #image_size = 32 x 32 x 3
# train = unpickle('cifar-100/train')
# # print(train.keys())
# # print(train[b'fine_labels'][20:3000])
# # raise False
# train_x = []
# train_x = np.array(train[b'data'].reshape((50000,32,32,3)))
# train_y = np.array(train[b'fine_labels'])
# train_y = keras.utils.to_categorical(train_y, num_classes=100)
#
# test = unpickle('cifar-100/test')
# test_x = np.array(test[b'data'].reshape((10000,32,32,3)))
# test_y = np.array(test[b'fine_labels'])
# test_y = keras.utils.to_categorical(test_y, num_classes=100)
# #print(test_y.shape)

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