# %%

#@title Import relevant modules
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth=200)

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# %%
x_train[2917]
# %%
plt.imshow(x_train[2917])
# %%

x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0


# %%
def plot_curve(epochs, hist, metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    plt.legend()


# %%
def create_model(learning_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(units=32, activation=keras.activations.relu))
    model.add(keras.layers.Dropout(rate=.2))
    model.add(
        keras.layers.Dense(units=10, activation=keras.activations.softmax))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.Accuracy()])
    return model


# %%
def train_model(model: keras.Model,
                train_features,
                train_label,
                epochs: int,
                batch_size: int = None,
                validation_split: float = 0.1):
    history = model.fit(
        x=train_features,
        y=train_label,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split=validation_split,
    )
    epoch = history.epoch
    hist = pd.DataFrame(history.history)
    return epoch, hist


# %%

learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

my_model = create_model(learning_rate)
my_model.summary()

print(x_train_normalized.shape)
print(y_train.shape)

epoch, hist = train_model(my_model, x_train_normalized, y_train, epochs,
                          batch_size, validation_split)

plot_curve(epoch, hist, ["accuracy"])

# Evaluate against the test set.
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)
# %%
