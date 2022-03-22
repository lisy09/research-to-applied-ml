# %%
from numpy import gradient
import tensorflow as tf
import tensorflow.keras as keras

# %% Import mnist data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.

# add a channel dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

# %% shuffle data and batch
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# %% keras model subclassing API to create model
class MyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            activation=keras.activations.relu
        )
        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(units=128, activation=keras.activations.relu)
        self.d2 = keras.layers.Dense(units=10)
    
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

model = MyModel()

# %% 
loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()

# %% 
train_loss = keras.metrics.Mean(name="train_loss")
train_accuracy = keras.metrics.SparseCategoricalAccuracy(name="train_acc")

test_loss = keras.metrics.Mean(name="test_loss")
test_accuracy = keras.metrics.SparseCategoricalAccuracy(name="test_acc")
# %% define training step
@tf.function
def train_step(images, label):
    with tf.GradientTape() as tape:
        # training = True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        prediction = model(images, training=True)
        loss = loss_func(label, prediction)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(label, prediction)
# %%


@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_func(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

# %%
EPOCHS = 5

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
# %%
