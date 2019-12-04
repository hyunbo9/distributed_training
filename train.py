import data
import vgg19
import time
import tensorflow as tf

model_syn = vgg19.vgg19_syn
model_asy = vgg19.vgg19_asy
train_x = data.train_x
train_y = data.train_y
test_x = data.test_x
test_y = data.test_y


#strategy = tf.distribute.MirroredStrategy()
#print('장치의 수: {}'.format(strategy.num_replicas_in_sync))

# BUFFER_SIZE = 10000
#
# BATCH_SIZE_PER_REPLICA = 64
#BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

#raise False

def input_fn(images, labels, epochs, batch_size):
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.repeat(epochs).batch(batch_size)
    return data



epochs = 30
batch_size = 400 * 8

syn = True

time1 = time.time()
if syn :
    model_syn.fit(train_x, train_y, epochs=epochs, batch_size = batch_size)
    test_loss, test_acc = model_syn.evaluate(test_x,  test_y, verbose=2, batch_size = 1000) # test는 size 1000으로 고정
else:

    model_asy.train(lambda: input_fn(train_x,
                                     train_y,
                                     epochs=epochs,
                                     batch_size=batch_size))

    acc = model_asy.evaluate(lambda: input_fn(test_x,
                                     test_y,
                                     epochs=epochs,
                                     batch_size=1000))  # test는 size 1000으로 고정
    print("acc", acc)

print("총 걸린 시간 :", time.time() - time1)
