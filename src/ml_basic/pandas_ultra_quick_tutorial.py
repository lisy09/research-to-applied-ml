import numpy as np
import pandas as pd

my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])
my_cols = ['temperature', 'activity']
my_df = pd.DataFrame(data=my_data, columns=my_cols)

print(my_df)

# add new col
my_df['adjusted'] = my_df['activity'] + 2
print(my_df)

# subset of df
print("First 3 rows:")
print(my_df[:3])
print(my_df.head(3))

print("Row #2")
print(my_df.iloc[[2]], '\n')

print("Col temperature:")
print(my_df['temperature'])

# Create a DataFrame
# Do the following:
# Create an 3x4 (3 rows x 4 columns) pandas DataFrame in which the columns are named Eleanor, Chidi, Tahani, and Jason.
# Populate each of the 12 cells in the DataFrame with a random integer between 0 and 100, inclusive.
# Output the following:
# the entire DataFrame
# the value in the cell of row #1 of the Eleanor column
# Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason.
# To complete this task, it helps to know the NumPy basics covered in the NumPy UltraQuick Tutorial.

cols = ['Eleanor', 'Chidi', 'Tahani', 'Jason']
data = np.random.randint(low=0, high=100, size=(3, 4))
df = pd.DataFrame(data=data, columns=cols)
print("entire df:")
print(df)
print("row #1 of the Eleanor col")
print(df['Eleanor'][1])
df['Janet'] = df['Tahani'] + df['Jason']
print(df, "\n")

# Create a reference by assigning my_dataframe to a new variable.
print("Experiment with a reference:")
reference_to_df = df

# Print the starting value of a particular cell.
print("  Starting value of df: %d" % df['Jason'][1])
print("  Starting value of reference_to_df: %d\n" %
      reference_to_df['Jason'][1])

# Modify a cell in df.
df.at[1, 'Jason'] = df['Jason'][1] + 5
print("  Updated df: %d" % df['Jason'][1])
print("  Updated reference_to_df: %d\n\n" % reference_to_df['Jason'][1])

# Create a true copy of my_dataframe
print("Experiment with a true copy:")
copy_of_my_df = my_df.copy()

# Print the starting value of a particular cell.
print("  Starting value of my_df: %d" % my_df['activity'][1])
print("  Starting value of copy_of_my_df: %d\n" % copy_of_my_df['activity'][1])

# Modify a cell in df.
my_df.at[1, 'activity'] = my_df['activity'][1] + 3
print("  Updated my_df: %d" % my_df['activity'][1])
print("  copy_of_my_df does not get updated: %d" %
      copy_of_my_df['activity'][1])
