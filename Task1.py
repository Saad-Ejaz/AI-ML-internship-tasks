# Load the dataset
import pandas as pd

df = pd.read_csv('Iris.csv')

# Print the shape, column names, and first few rows
print("Shape: " , df.shape)
print("Columns: ", df.columns.tolist())
print("First 5 rows: ", df.head())

# info
print("\n dataset info: ")
df.info()

print("\n Summary and Statistics: ", df.describe())

# visualization

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='Species')
plt.show()

df.hist(figsize=(10,8),bins=20,color='skyblue', edgecolor='black')
plt.suptitle('Histogram of Numerical Features')
plt.tight_layout()
plt.show()