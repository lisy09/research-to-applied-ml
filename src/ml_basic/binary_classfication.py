# %%
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
from matplotlib import pyplot as plt, units

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
# tf.keras.backend.set_floatx('float32')

print("Ran the import statements.")
# %%
train_df: pd.DataFrame = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
)
test_df: pd.DataFrame = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv"
)
train_df: pd.DataFrame = train_df.reindex(np.random.permutation(
    train_df.index))  # shuffle the training set

# %%
# When creating a model with multiple features, the values of each feature should cover roughly the same range. For example, if one feature's range spans 500 to 100,000 and another feature's range spans 2 to 12, then the model will be difficult or impossible to train. Therefore, you should normalize features in a multi-feature model.

# The following code cell normalizes datasets by converting each raw value (including the label) to its Z-score. A Z-score is the number of standard deviations from the mean for a particular raw value. For example, consider a feature having the following characteristics:

# The mean is 60.
# The standard deviation is 10.

# calculate z-scores of each col
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean) / train_df_std
train_df_norm.head()

# %%%
# write those Z-scores into a new pandas DataFrame named test_df_norm.
test_df_mean = test_df.mean()
test_df_std = test_df.std()
test_df_norm = (test_df - test_df_mean) / test_df_std
# %%
threshold = 265000
train_df_norm["median_house_value_is_high"] = (train_df["median_house_value"] >
                                               threshold).astype(float)
test_df_norm["median_house_value_is_high"] = (test_df["median_house_value"] >
                                              threshold).astype(float)
train_df_norm["median_house_value_is_high"].head(8000)
# %%

feature_cols = []

feature_cols.append(tf.feature_column.numeric_column("median_income"))
feature_cols.append(tf.feature_column.numeric_column("total_rooms"))

feature_layer = layers.DenseFeatures(feature_cols)

# print the first 3 and last 3 rows of the feature_layers output when applied to train_df_norm
feature_layer(dict(train_df_norm))
# %%


def create_model(learning_rate: float, feature_layer: layers.DenseFeatures,
                 metrics: List[keras.metrics.Metric]):
    model = keras.models.Sequential()
    model.add(feature_layer)
    model.add(layers.Dense(units=1, input_shape=(1, ), activation=tf.sigmoid))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)
    return model


def train_model(model: keras.models.Model,
                dataset: pd.DataFrame,
                epochs: int,
                label_name: str,
                batch_size: int,
                shuffle: bool = True):
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(
        x=features,
        y=label,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle,
    )
    epoch = history.epoch
    hist = pd.DataFrame(history.history)
    return epoch, hist


# %%
#@title Define the plotting function.
def plot_curve(epochs, hist, list_of_metrics):
    """Plot a curve of one or more classification metrics vs. epoch."""
    # list_of_metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()


print("Defined the plot_curve function.")
# %%
# The following variables are the hyperparameters.
learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = "median_house_value_is_high"
classification_threshold = 0.35

# Establish the metrics the model will measure.
METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy',
                                 threshold=classification_threshold),
]

# Establish the model's topography.
my_model = create_model(learning_rate, feature_layer, METRICS)

# Train the model on the training set.
epochs, hist = train_model(my_model, train_df_norm, epochs, label_name,
                           batch_size)

# Plot a graph of the metric(s) vs. epochs.
list_of_metrics_to_plot = ['accuracy']

plot_curve(epochs, hist, list_of_metrics_to_plot)
# %%

features = {name: np.array(value) for name, value in test_df_norm.items()}
label = np.array(features.pop(label_name))

my_model.evaluate(x=features, y=label, batch_size=batch_size)
# %%
# add precision and recall as metrics

learning_rate = 0.001
epochs = 20
batch_size = 100
classification_threshold = 0.35
label_name = "median_house_value_is_high"

# Modify the following definition of METRICS to generate
# not only accuracy and precision, but also recall:
METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy',
                                 threshold=classification_threshold),
    keras.metrics.Precision(thresholds=classification_threshold,
                            name='precision'),
    keras.metrics.Recall(thresholds=classification_threshold, name="recall")
]

# Establish the model's topography.
my_model = create_model(learning_rate, feature_layer, METRICS)

# Train the model on the training set.
epochs, hist = train_model(my_model, train_df_norm, epochs, label_name,
                           batch_size)

# Plot metrics vs. epochs
list_of_metrics_to_plot = ['accuracy', 'precision', 'recall']
plot_curve(epochs, hist, list_of_metrics_to_plot)
# %%
#@title Double-click to view the solution for Task 4.

# The following variables are the hyperparameters.
learning_rate = 0.001
epochs = 20
batch_size = 100
classification_threshold = 0.52
label_name = "median_house_value_is_high"

# Here is the updated definition of METRICS:
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy',
                                    threshold=classification_threshold),
    tf.keras.metrics.Precision(thresholds=classification_threshold,
                               name='precision'),
    tf.keras.metrics.Recall(thresholds=classification_threshold,
                            name="recall"),
]

# Establish the model's topography.
my_model = create_model(learning_rate, feature_layer, METRICS)

# Train the model on the training set.
epochs, hist = train_model(my_model, train_df_norm, epochs, label_name,
                           batch_size)

# Plot metrics vs. epochs
list_of_metrics_to_plot = ['accuracy', "precision", "recall"]
plot_curve(epochs, hist, list_of_metrics_to_plot)

# A `classification_threshold` of slightly over 0.5
# appears to produce the highest accuracy (about 83%).
# Raising the `classification_threshold` to 0.9 drops
# accuracy by about 5%.  Lowering the
# `classification_threshold` to 0.3 drops accuracy by
# about 3%.
# %%
# The following variables are the hyperparameters.
learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = "median_house_value_is_high"

# AUC is a reasonable "summary" metric for
# classification models.
# Here is the updated definition of METRICS to
# measure AUC:
METRICS = [
    tf.keras.metrics.AUC(num_thresholds=100, name='auc'),
]

# Establish the model's topography.
my_model = create_model(learning_rate, feature_layer, METRICS)

# Train the model on the training set.
epochs, hist = train_model(my_model, train_df_norm, epochs, label_name,
                           batch_size)

# Plot metrics vs. epochs
list_of_metrics_to_plot = ['auc']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# %%
