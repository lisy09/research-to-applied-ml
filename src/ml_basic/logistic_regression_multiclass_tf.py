# %%
from cProfile import run
import enum
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
# %%

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

x_train, x_test = x_train.reshape([-1, 28**2]), x_test.reshape([-1, 28**2])
x_train, x_test = x_train / 255., x_test / 255.
# %%
num_classes = 10  # 0 to 9 digits

num_features = 28**2

# Training parameters.

learning_rate = 0.01

training_steps = 1000

batch_size = 256

display_step = 50
# %%
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
# %%
W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
b = tf.Variable(tf.zeros([num_classes]), name="bias")


# %%
def lr(x: tf.Tensor, w: tf.Tensor, b: tf.Tensor):
    return tf.nn.softmax(tf.matmul(x, w) + b)


# %%
def cross_entropy(y_pred: tf.Tensor, y_true: tf.Tensor, num_classes: int):
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1)
    return tf.reduce_mean(-tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred)))


# %%
def accuracy(y_pred: tf.Tensor, y_true: tf.Tensor):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1),
                                  tf.cast(y_true, tf.int64))
    return tf.reduce_sum(tf.cast(correct_prediction, tf.float32))


# %%
optimizer = keras.optimizers.SGD(learning_rate)


def run_optimization(x: tf.Tensor, y: tf.Tensor):
    with tf.GradientTape() as g:
        pred = lr(x, W, b)
        loss = cross_entropy(pred, y, num_classes)

    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


# %%
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)
    if step % display_step == 0:
        pred = lr(batch_x, W, b)
        loss = cross_entropy(pred, batch_y, num_classes)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %%
pred = lr(x_test, W, b)

print("Test Accuracy: %f" % accuracy(pred, y_test))
# %%
