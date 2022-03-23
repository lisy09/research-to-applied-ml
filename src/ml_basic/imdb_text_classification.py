# %%%
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

# %%
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = keras.utils.get_file("aclImdb_v1",
                               url,
                               untar=True,
                               cache_dir='.',
                               cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

# %%
os.listdir(dataset_dir)
# %%
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

# %%
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())

# %%
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# %%
batch_size = 32
seed = 42

raw_train_ds = keras.utils.text_dataset_from_directory(
    directory='aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

# %%
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

# %%
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

# %%
raw_val_ds = keras.utils.text_dataset_from_directory('aclImdb/train',
                                                     batch_size=batch_size,
                                                     validation_split=0.2,
                                                     subset='validation',
                                                     seed=seed)

# %%
raw_test_ds = keras.utils.text_dataset_from_directory('aclImdb/test',
                                                      batch_size=batch_size)


# %%
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    f'[{re.escape(string.punctuation)}]', '')


# %%
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# %%
# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


# %%
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# %%
# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

# %%
print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# %%
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# %%
# These are two important methods you should use when loading data to make sure that I/O does not become blocking.
# .cache() keeps data in memory after it's loaded off disk. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache, which is more efficient to read than many small files.
# .prefetch() overlaps data preprocessing and model execution while training.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %%
embedding_dim = 16

# %%
model = keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
])
model.summary()
# %%
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=keras.metrics.BinaryAccuracy(threshold=0.0))
# %%
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# %%
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
# %%
history_dict = history.history
history_dict.keys()
# %%
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
# %%
export_model = tf.keras.Sequential(
    [vectorize_layer, model,
     layers.Activation('sigmoid')])

export_model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
                     optimizer="adam",
                     metrics=['accuracy'])

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)
# %%
examples = [
    "The movie was great!", "The movie was okay.", "The movie was terrible..."
]

export_model.predict(examples)
# %%
