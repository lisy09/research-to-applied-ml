# %%

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
import tensorflow.keras as keras

from matplotlib import pyplot as plt, units

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

tf.keras.backend.set_floatx('float32')

print("Imported the modules.")
# %%
# Load the dataset
train_df = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
)
test_df = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv"
)

# Scale the labels
scale_factor = 1000.0
# Scale the training set's label.
train_df["median_house_value"] /= scale_factor

# Scale the test set's label
test_df["median_house_value"] /= scale_factor

# Shuffle the examples
train_df = train_df.reindex(np.random.permutation(train_df.index))
# %%
feature_cols = []

feature_cols.append(tf.feature_column.numeric_column("latitude"))
feature_cols.append(tf.feature_column.numeric_column("longitude"))

float_feature_layer = keras.layers.DenseFeatures(feature_cols)


# %%
def create_model(learning_rate, feature_layer: keras.layers.Layer):
    model = keras.models.Sequential()
    model.add(feature_layer)
    model.add(keras.layers.Dense(units=1, input_shape=(1, )))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            keras.metrics.RootMeanSquaredError(),
        ])

    return model


def train_model(model: keras.Model, dataset: pd.DataFrame, epochs, batch_size,
                label_name):
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True)
    epochs = history.epoch
    rmse = pd.DataFrame(history.history)['root_mean_squared_error']
    return epochs, rmse


def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.94, rmse.max() * 1.05])
    plt.show()


print(
    "Defined the create_model, train_model, and plot_the_loss_curve functions."
)
# %%
# The following variables are the hyperparameters.
learning_rate = 0.05
epochs = 30
batch_size = 100
label_name = 'median_house_value'

# Create and compile the model's topography.
my_model = create_model(learning_rate, float_feature_layer)

# Train the model on the training set.
epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, rmse)

print("\n: Evaluate the new model against the test set:")
test_features = {name: np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name))
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)
# %%

# Are floating-point values a good way to represent latitude and longitude?

# No. Representing latitude and longitude as
# floating-point values does not have much
# predictive power. For example, neighborhoods at
# latitude 35 are not 36/35 more valuable
# (or 35/36 less valuable) than houses at
# latitude 36.

# Representing `latitude` and `longitude` as
# floating-point values provides almost no
# predictive power. We're only using the raw values
# to establish a baseline for future experiments
# with better representations.
# %%

# Represent latitude and longitude in buckets
# The following code cell represents latitude and longitude in buckets (bins). Each bin represents all the neighborhoods within a single degree. For example, neighborhoods at latitude 35.4 and 35.8 are in the same bucket, but neighborhoods in latitude 35.4 and 36.2 are in different buckets.

# The model will learn a separate weight for each bucket. For example, the model will learn one weight for all the neighborhoods in the "35" bin", a different weight for neighborhoods in the "36" bin, and so on. This representation will create approximately 20 buckets:

# 10 buckets for latitude.
# 10 buckets for longitude.

resolution_in_degrees = 1.0

# Create a new empty list that will eventually hold the generated feature column.
feature_columns = []

# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(
    np.arange(int(min(train_df['latitude'])), int(max(train_df['latitude'])),
              resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,
                                               latitude_boundaries)
feature_columns.append(latitude)

# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(
    np.arange(int(min(train_df['longitude'])), int(max(train_df['longitude'])),
              resolution_in_degrees))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                                longitude_boundaries)
feature_columns.append(longitude)

# Convert the list of feature columns into a layer that will ultimately become
# part of the model. Understanding layers is not important right now.
buckets_feature_layer = keras.layers.DenseFeatures(feature_columns)

# The following variables are the hyperparameters.
learning_rate = 0.04
epochs = 35

# Build the model, this time passing in the buckets_feature_layer.
my_model = create_model(learning_rate, buckets_feature_layer)

# Train the model on the training set.
epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, rmse)

print("\n: Evaluate the new model against the test set:")
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)
# %%

# Represent location as a feature cross
# The following code cell represents location as a feature cross. That is, the following code cell first creates buckets and then calls tf.feature_column.crossed_column to cross the buckets.

resolution_in_degrees = 1.0

# Create a new empty list that will eventually hold the generated feature column.
feature_columns = []

# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(
    np.arange(int(min(train_df['latitude'])), int(max(train_df['latitude'])),
              resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,
                                               latitude_boundaries)

# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(
    np.arange(int(min(train_df['longitude'])), int(max(train_df['longitude'])),
              resolution_in_degrees))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                                longitude_boundaries)

# Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude],
                                                        hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Convert the list of feature columns into a layer that will later be fed into
# the model.
feature_cross_feature_layer = keras.layers.DenseFeatures(feature_columns)

# %%
# The following variables are the hyperparameters.
learning_rate = 0.04
epochs = 35

# Build the model, this time passing in the feature_cross_feature_layer: 
my_model = create_model(learning_rate, feature_cross_feature_layer)

# Train the model on the training set.
epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, rmse)

print("\n: Evaluate the new model against the test set:")
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)
# %%
