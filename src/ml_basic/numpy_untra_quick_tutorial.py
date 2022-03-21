import numpy as np

array_1d = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(array_1d)

array_2d = np.array([[6, 5], [11, 7], [4, 8]])
print(array_2d)

# populate arrays with sequences of number
array_seq = np.arange(5, 12)
print(array_seq)

# populate with random numbers
array_random_int_50_to_100 = np.random.randint(low=50, high=101, size=(6))
print(array_random_int_50_to_100)

array_random_float_0_to_1 = np.random.random((6))
print(array_random_float_0_to_1)

# math op
# NumPy uses a trick called broadcasting to virtually expand the smaller operand to dimensions compatible for linear algebra
random_floats_2_to_3 = np.random.random((6)) + 2
print('random_floats_2_to_3', random_floats_2_to_3)

# Create a Linear Dataset
# Your goal is to create a simple dataset consisting of a single feature and a label as follows:
# Assign a sequence of integers from 6 to 20 (inclusive) to a NumPy array named feature.
# Assign 15 values to a NumPy array named label such that:
#    label = (3)(feature) + 4
# For example, the first value for label should be:
#   label = (3)(6) + 4 = 22

features = np.arange(6, 21)
print("features", features)
labels = features * 3 + 4
print("labels", labels)

# Add Some Noise to the Dataset
# To make your dataset a little more realistic, insert a little random noise into each element
# of the label array you already created. To be more precise, modify each value assigned to label
# by adding a different random floating-point value between -2 and +2.
# Don't rely on broadcasting. Instead, create a noise array having the same dimension as label.
noise = np.random.random(size=labels.size) * 4 - 2
print('noise', noise)
noised_label = noise + labels
print('noised_label', noised_label)
