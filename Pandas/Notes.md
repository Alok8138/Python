# NumPy + Pandas Cheat Sheet for Machine Learning

## Import Libraries

```python
import numpy as np   # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation
```

---

# NUMPY

NumPy is used for numerical computing and working with arrays. Almost all ML libraries internally use NumPy arrays.

## Creating Arrays

```python
np.array([1,2,3])           # Create a 1D array from a list
np.array([[1,2],[3,4]])     # Create a 2D array (matrix) from nested lists
np.zeros((2,3))             # Create a 2x3 matrix filled with 0s
np.ones((2,3))              # Create a 2x3 matrix filled with 1s
np.arange(0,10)             # Create array [0,1,2,3,4,5,6,7,8,9]
```

## Array Information

```python
a.shape    # Returns (rows, columns) — dimensions of the array
a.ndim     # Returns number of dimensions (1D, 2D, etc.)
a.size     # Returns total number of elements in the array
a.dtype    # Returns data type of elements (int, float, etc.)
```

Example:

```python
a = np.array([[1,2,3],[4,5,6]])

a.shape    # Returns (2, 3) — 2 rows and 3 columns
```

## Indexing

```python
a[0]       # Access first element (or first row in 2D)
a[1]       # Access second element (or second row in 2D)
a[1,0]     # Access element at row 1, column 0
a[0,2]     # Access element at row 0, column 2
```

## Slicing

```python
a[1:4]     # Get elements from index 1 to 3 (end index is excluded)
a[:3]      # Get first 3 elements (index 0, 1, 2)
a[2:]      # Get all elements from index 2 to end
a[:,1]     # Get entire second column (all rows, column 1)
a[0,:]     # Get entire first row (row 0, all columns)
```

## Reshaping Arrays

```python
a.reshape(2,3)   # Reshape array into 2 rows and 3 columns
a.reshape(3,2)   # Reshape array into 3 rows and 2 columns
a.flatten()      # Flatten any array into a 1D array
```

## Mathematical Operations

```python
a + 5    # Add 5 to every element
a - 2    # Subtract 2 from every element
a * 2    # Multiply every element by 2
a / 2    # Divide every element by 2
a ** 2   # Square every element
```

## Array vs Array Operations

```python
a + b    # Add corresponding elements of two arrays
a - b    # Subtract corresponding elements
a * b    # Multiply corresponding elements (not matrix multiplication)
a / b    # Divide corresponding elements
```

## Aggregation Functions

```python
a.sum()    # Returns sum of all elements
a.mean()   # Returns average of all elements
a.max()    # Returns the largest value in the array
a.min()    # Returns the smallest value in the array
a.std()    # Returns standard deviation of elements
a.var()    # Returns variance of elements
```

## Matrix Multiplication

```python
A @ B        # Multiply two matrices (dot product)
np.dot(A,B)  # Same as above — multiply two matrices
```

## Transpose

```python
A.T    # Flip the matrix — rows become columns, columns become rows
```

Example:

```python
[[1,2,3],    # Before transpose
 [4,5,6]]

[[1,4],      # After transpose — rows became columns
 [2,5],
 [3,6]]
```

## Random Numbers

```python
np.random.rand(3)           # Generate 3 random float values between 0 and 1
np.random.randint(1,10,size=5)  # Generate 5 random integers between 1 and 9
np.random.seed(42)          # Fix random seed so results are reproducible
```

## Useful ML Functions

```python
a.argmax()      # Returns index of the largest value in the array
a.argmin()      # Returns index of the smallest value in the array
np.unique(a)    # Returns sorted array of unique (non-duplicate) values
np.where(a>5)   # Returns indices of elements that satisfy the condition
```

---

# PANDAS

Pandas is used for loading, cleaning, filtering, and analyzing datasets.

Think of it as Excel inside Python.

## Reading Dataset

```python
df = pd.read_csv("data.csv")   # Load a CSV file into a DataFrame
```

## Viewing Dataset

```python
df.head()      # Show first 5 rows of the DataFrame
df.tail()      # Show last 5 rows of the DataFrame
df.sample(5)   # Show 5 randomly selected rows
```

## Dataset Information

```python
df.info()       # Show column names, data types, and non-null counts
df.describe()   # Show statistics like mean, min, max for numeric columns
df.shape        # Returns (number of rows, number of columns)
df.columns      # Returns list of all column names
df.dtypes       # Returns data type of each column
```

## Selecting Columns

```python
df["Age"]             # Select single column — returns a Series
df[["Name","Age"]]    # Select multiple columns — returns a DataFrame
```

## Selecting Rows

### By Position

```python
df.iloc[0]      # Select row at position 0 (first row)
df.iloc[0:5]    # Select rows from position 0 to 4
```

### By Label

```python
df.loc[0]       # Select row with index label 0
df.loc[0:5]     # Select rows with index labels 0 to 5 (end inclusive)
```

## Filtering Data

```python
df[df["Age"] > 22]                              # Keep rows where Age is greater than 22
df[df["Salary"] > 50000]                        # Keep rows where Salary is greater than 50000
df[(df["Age"] > 22) & (df["Salary"] > 50000)]  # Keep rows satisfying both conditions
```

## Creating New Columns

```python
df["Bonus"] = df["Salary"] * 0.1   # Create new column Bonus = 10% of Salary
```

## Removing Columns

```python
df.drop("Bonus", axis=1)   # Remove the Bonus column (axis=1 means column)
```

## Missing Values

### Check Missing Values

```python
df.isnull().sum()   # Count number of missing (NaN) values in each column
```

### Remove Missing Values

```python
df.dropna()   # Remove all rows that have at least one missing value
```

### Fill Missing Values

```python
df.fillna(0)                        # Replace all missing values with 0
df["Age"].fillna(df["Age"].mean())  # Replace missing Age values with the column mean
```

## Sorting

```python
df.sort_values("Age")                    # Sort rows by Age in ascending order
df.sort_values("Age", ascending=False)   # Sort rows by Age in descending order
```

## Unique Values

```python
df["Department"].unique()    # Return array of all unique values in Department column
df["Department"].nunique()   # Return count of unique values in Department column
```

## Count Occurrences

```python
df["Department"].value_counts()   # Count how many times each Department value appears
```

## GroupBy

```python
df.groupby("Department")["Salary"].mean()   # Group by Department and get average Salary per group
```

Other common aggregations:

```python
.sum()     # Total of each group
.max()     # Maximum value in each group
.min()     # Minimum value in each group
.count()   # Number of rows in each group
```

## Merging DataFrames

```python
pd.merge(df1, df2, on="ID")   # Join df1 and df2 on the common column ID (like SQL JOIN)
```

## Saving Dataset

```python
df.to_csv("output.csv", index=False)   # Save DataFrame to CSV, index=False skips row numbers
```

---

# Typical ML Workflow

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")   # Load the dataset

df.head()       # Preview first 5 rows
df.info()       # Check column types and missing values
df.describe()   # Get statistical summary

df.fillna(0)    # Fill missing values with 0

X = df[["Age","Salary"]]   # Select input features
y = df["Purchased"]        # Select target column

X = X.values   # Convert features DataFrame to NumPy array
y = y.values   # Convert target Series to NumPy array

X.shape   # Check shape of features array
y.shape   # Check shape of target array
```

---

# Functions You Must Memorize

## NumPy

```python
array()           # Create an array from a list
zeros()           # Create array filled with 0s
ones()            # Create array filled with 1s
arange()          # Create array with a range of values
shape             # Get dimensions of the array
reshape()         # Change shape without changing data
flatten()         # Convert to 1D array
sum()             # Sum of all elements
mean()            # Average of all elements
max()             # Largest value
min()             # Smallest value
std()             # Standard deviation
var()             # Variance
dot()             # Matrix multiplication
T                 # Transpose the matrix
argmax()          # Index of largest value
argmin()          # Index of smallest value
unique()          # Get unique values
where()           # Get indices where condition is True
random.rand()     # Random floats between 0 and 1
random.randint()  # Random integers in a range
```

## Pandas

```python
read_csv()       # Load CSV file into DataFrame
head()           # Show first 5 rows
tail()           # Show last 5 rows
sample()         # Show random rows
info()           # Show column info and null counts
describe()       # Show summary statistics
shape            # Get (rows, columns)
columns          # Get column names
dtypes           # Get data types of columns
iloc[]           # Select rows by position
loc[]            # Select rows by label
isnull()         # Check for missing values
fillna()         # Fill missing values
dropna()         # Drop rows with missing values
sort_values()    # Sort rows by column
groupby()        # Group rows and aggregate
value_counts()   # Count occurrences of each value
unique()         # Get unique values in a column
nunique()        # Count unique values in a column
merge()          # Join two DataFrames
to_csv()         # Save DataFrame to CSV
```

---

# After Learning This

Next topics:

1. Matplotlib Basics
2. Seaborn Basics
3. Scikit-Learn
4. Machine Learning Algorithms
5. Andrew Ng ML Specialization

If you are comfortable with everything in this cheat sheet, you already know about 90% of the NumPy and Pandas needed for beginner-to-intermediate Machine Learning.
