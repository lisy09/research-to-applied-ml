# %%
from cProfile import label
import imp
from turtle import shape
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from matplotlib import pyplot as plt, units

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# %% Load datasets from Internet

train_df: pd.DataFrame = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
)
test_df: pd.DataFrame = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv"
)

# %%
scale_label = 1000.
train_df['median_house_value'] /= scale_label
test_df['median_house_value'] /= scale_label


# %%
def build_model(learning_rate: float):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(1, )))
    model.add(keras.layers.Dense(units=1))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.RootMeanSquaredError()])
    return model


def train_model(model: keras.Model,
                df: pd.DataFrame,
                feature,
                label,
                epochs: int,
                batch_size=None,
                validation_split=0.1):
    history = model.fit(
        x=df[feature],
        y=df[label],
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
    )

    params = model.get_weights()
    weight = params[0]
    bias = params[1]

    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return history.epoch, rmse, history.history


# %%
def plot_loss_curve(epochs, mae_training, mae_validation):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")

    plt.plot(epochs[1:], mae_training[1:], label="training loss")
    plt.plot(epochs[1:], mae_validation[1:], label="validation loss")
    plt.legend()

    # We're not going to plot the first epoch, since the loss on the first epoch
    # is often substantially greater than the loss for other epochs.
    merged_mae_lists = mae_training[1:] + mae_validation[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)

    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.show()


# %%
learning_rate = 0.05
epoch = 30
batch_size = 100
validation_split = 0.2

my_feature = "median_income"
my_label = "median_house_value"

my_model = None
my_model = build_model(learning_rate)

epochs, rmse, history = train_model(my_model, train_df, my_feature, my_label,
                                    epoch, batch_size, validation_split)

plot_loss_curve(epochs, history["root_mean_squared_error"],
                history["val_root_mean_squared_error"])

# %%

# No matter how you split the training set and the validation set, the loss curves differ significantly. Evidently, the data in the training set isn't similar enough to the data in the validation set. Counterintuitive? Yes, but this problem is actually pretty common in machine learning.
# Your task is to determine why the loss curves aren't highly similar. As with most issues in machine learning, the problem is rooted in the data itself. To solve this mystery of why the training set and validation set aren't almost identical, write a line or two of pandas code in the following code cell. Here are a couple of hints:
# The previous code cell split the original training set into:
# a reduced training set (the original training set - the validation set)
# the validation set
# By default, the pandas head method outputs the first 5 rows of the DataFrame. To see more of the training set, specify the n argument to head and assign a large positive integer to n.

train_df.head(1000)
# %%
# shuffle data before splitting
shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))
epochs, rmse, history = train_model(my_model, shuffled_train_df, my_feature,
                                    my_label, epoch, batch_size,
                                    validation_split)

plot_loss_curve(epochs, history["root_mean_squared_error"],
                history["val_root_mean_squared_error"])
# %%
