# NumPy + Pandas Cheat Sheet for Machine Learning

## Import Libraries

```python
import numpy as np
import pandas as pd
```

---

# NUMPY

NumPy is used for numerical computing and working with arrays. Almost all ML libraries internally use NumPy arrays.

## Creating Arrays

```python
np.array([1,2,3])              # 1D array

np.array([[1,2],[3,4]])        # 2D array

np.zeros((2,3))                # Matrix of zeros

np.ones((2,3))                 # Matrix of ones

np.arange(0,10)                # [0,1,2,3,4,5,6,7,8,9]
```

## Array Information

```python
a.shape      # Dimensions (rows, columns)

a.ndim       # Number of dimensions

a.size       # Total elements

a.dtype      # Data type
```

Example:

```python
a = np.array([[1,2,3],[4,5,6]])

a.shape
# (2,3)
```

Meaning:

- 2 rows
- 3 columns

## Indexing

```python
a[0]         # First element

a[1]         # Second element

a[1,0]       # Row 1 Column 0

a[0,2]       # Row 0 Column 2
```

## Slicing

```python
a[1:4]       # Elements from index 1 to 3

a[:3]        # First 3 elements

a[2:]        # From index 2 to end

a[:,1]       # Entire second column

a[0,:]       # Entire first row
```

## Reshaping Arrays

```python
a.reshape(2,3)

a.reshape(3,2)

a.flatten()      # Convert to 1D array
```

Very common in ML preprocessing.

## Mathematical Operations

```python
a + 5

a - 2

a * 2

a / 2

a ** 2
```

Operations are applied to every element.

## Array vs Array Operations

```python
a + b

a - b

a * b

a / b
```

## Aggregation Functions

```python
a.sum()      # Sum of all elements

a.mean()     # Average

a.max()      # Maximum

a.min()      # Minimum

a.std()      # Standard deviation

a.var()      # Variance
```

Used constantly during data analysis.

## Matrix Multiplication

```python
A @ B

np.dot(A,B)
```

Machine Learning is heavily based on matrix multiplication.

## Transpose

```python
A.T
```

Converts rows into columns and vice versa.

Example:

```python
[[1,2,3],
 [4,5,6]]
```

becomes

```python
[[1,4],
 [2,5],
 [3,6]]
```

## Random Numbers

```python
np.random.rand(3)

np.random.randint(1,10,size=5)

np.random.seed(42)
```

Used for initializing models and generating random samples.

## Useful ML Functions

```python
a.argmax()      # Index of largest value

a.argmin()      # Index of smallest value

np.unique(a)    # Unique values

np.where(a>5)   # Indices satisfying condition
```

---

# PANDAS

Pandas is used for loading, cleaning, filtering, and analyzing datasets.

Think of it as Excel inside Python.

## Reading Dataset

```python
df = pd.read_csv("data.csv")
```

Most datasets are CSV files.

## Viewing Dataset

```python
df.head()       # First 5 rows

df.tail()       # Last 5 rows

df.sample(5)    # Random 5 rows
```

## Dataset Information

```python
df.info()

df.describe()

df.shape

df.columns

df.dtypes
```

Useful for understanding a dataset quickly.

## Selecting Columns

```python
df["Age"]

df[["Name","Age"]]
```

## Selecting Rows

### By Position

```python
df.iloc[0]

df.iloc[0:5]
```

### By Label

```python
df.loc[0]

df.loc[0:5]
```

## Filtering Data

```python
df[df["Age"] > 22]

df[df["Salary"] > 50000]

df[(df["Age"] > 22) &
   (df["Salary"] > 50000)]
```

Very common during data cleaning.

## Creating New Columns

```python
df["Bonus"] = df["Salary"] * 0.1
```

## Removing Columns

```python
df.drop("Bonus", axis=1)
```

## Missing Values

### Check Missing Values

```python
df.isnull().sum()
```

### Remove Missing Values

```python
df.dropna()
```

### Fill Missing Values

```python
df.fillna(0)

df["Age"].fillna(df["Age"].mean())
```

One of the most important data preprocessing steps.

## Sorting

```python
df.sort_values("Age")

df.sort_values("Age", ascending=False)
```

## Unique Values

```python
df["Department"].unique()

df["Department"].nunique()
```

## Count Occurrences

```python
df["Department"].value_counts()
```

Very useful in exploratory data analysis (EDA).

## GroupBy

```python
df.groupby("Department")["Salary"].mean()
```

Other common aggregations:

```python
.sum()

.max()

.min()

.count()
```

## Merging DataFrames

```python
pd.merge(df1, df2, on="ID")
```

Similar to SQL JOIN.

## Saving Dataset

```python
df.to_csv("output.csv", index=False)
```

---

# Typical ML Workflow

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data.csv")

# Inspect data
df.head()
df.info()
df.describe()

# Handle missing values
df.fillna(0)

# Features (input)
X = df[["Age","Salary"]]

# Target (output)
y = df["Purchased"]

# Convert to NumPy arrays
X = X.values
y = y.values

# Check dimensions
X.shape
y.shape
```

---

# Functions You Must Memorize

## NumPy

```python
array()
zeros()
ones()
arange()
shape
reshape()
flatten()
sum()
mean()
max()
min()
std()
var()
dot()
T
argmax()
argmin()
unique()
where()
random.rand()
random.randint()
```

## Pandas

```python
read_csv()
head()
tail()
sample()
info()
describe()
shape
columns
dtypes
iloc[]
loc[]
isnull()
fillna()
dropna()
sort_values()
groupby()
value_counts()
unique()
nunique()
merge()
to_csv()
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
