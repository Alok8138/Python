# NumPy Complete Notes for Machine Learning

> 📌 **Goal:** This file is your **only** NumPy reference before starting Machine Learning.  
> Everything you need — from scratch to intermediate — is right here.

---

## Table of Contents

| # | Topic |
|---|-------|
| 1 | [Introduction](#1-introduction) |
| 2 | [Installation & Import](#2-installation--import) |
| 3 | [Creating Arrays](#3-creating-arrays) |
| 4 | [Array Basics](#4-array-basics) |
| 5 | [Indexing & Slicing](#5-indexing--slicing-very-important) |
| 6 | [Array Operations](#6-array-operations) |
| 7 | [Broadcasting](#7-broadcasting-critical) |
| 8 | [Mathematical Functions](#8-mathematical-functions) |
| 9 | [Reshaping & Transformations](#9-reshaping--transformations) |
| 10 | [Matrix Operations](#10-matrix-operations) |
| 11 | [Useful Functions for ML](#11-useful-functions-for-ml) |
| 12 | [Performance Advantage](#12-performance-advantage) |
| 13 | [Common Mistakes](#13-common-mistakes) |
| 14 | [Mini Practice Section](#14-mini-practice-section) |
| 15 | [Quick Revision Sheet](#15-quick-revision-sheet) |

---

## 1. Introduction

### What is NumPy?

**NumPy** (Numerical Python) is the foundational library for numerical computing in Python.  
It provides a powerful **N-dimensional array object** and tools for working with these arrays efficiently.

Think of it as the "engine" that powers almost every ML/Data Science library:
- **Pandas** is built on top of NumPy
- **Scikit-learn** uses NumPy arrays internally
- **TensorFlow / PyTorch** mimic NumPy's API

### Why NumPy is Important in ML

| Reason | Detail |
|--------|--------|
| 🚀 Speed | Operations run in C under the hood — 50–100x faster than Python loops |
| 🧮 Math | Built-in linear algebra, statistics, and matrix operations |
| 🔗 Compatibility | Every ML library accepts NumPy arrays as input |
| 📦 Memory | More compact storage than Python lists |
| 🎯 Vectorization | Eliminates the need for slow Python for-loops |

> In ML, your **data is always a matrix (2D array)**. NumPy is how you work with matrices in Python.

### Difference Between Python Lists and NumPy Arrays

```python
import numpy as np

# Python List
py_list = [1, 2, 3, 4, 5]

# NumPy Array
np_array = np.array([1, 2, 3, 4, 5])
```

| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| Data Types | Mixed (int, str, etc.) | Homogeneous (same type) |
| Speed | Slow | Very Fast |
| Memory | More | Less |
| Math Operations | Manual loop needed | Direct vectorized ops |
| Dimensions | 1D only (nesting = messy) | N-dimensional natively |
| Syntax | `[a+b for a,b in zip(l1,l2)]` | `arr1 + arr2` |

```python
# Lists require loops for math
list_a = [1, 2, 3]
list_b = [4, 5, 6]
result = [a * 2 for a in list_a]          # [2, 4, 6] — needs loop

# NumPy does it directly
arr = np.array([1, 2, 3])
result = arr * 2                           # [2, 4, 6] — no loop needed!
```

---

## 2. Installation & Import

### Installation

```bash
# Install NumPy using pip
pip install numpy

# Or if using Anaconda (already included by default)
conda install numpy
```

### Verify Installation

```python
import numpy as np
print(np.__version__)   # e.g., 1.26.4
```

### The Standard Import Convention

```python
import numpy as np   # 'np' is the universal alias — always use this
```

> ⚠️ **Always use `np` as the alias.** Every tutorial, book, and codebase uses this convention.

---

## 3. Creating Arrays

### 3.1 `np.array()` — From a Python List

```python
import numpy as np

# 1D Array (Vector)
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]

# 2D Array (Matrix) — list of lists
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]

# 3D Array — list of list of lists
c = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])
print(c.shape)    # (2, 2, 2)
```

---

### 3.2 `np.arange()` — Like Python's `range()`

```python
# Syntax: np.arange(start, stop, step)

a = np.arange(10)           # [0 1 2 3 4 5 6 7 8 9]
b = np.arange(1, 10)        # [1 2 3 4 5 6 7 8 9]
c = np.arange(0, 10, 2)     # [0 2 4 6 8]
d = np.arange(0, 1, 0.1)    # [0.  0.1  0.2 ... 0.9]
```

---

### 3.3 `np.linspace()` — Evenly Spaced Points

```python
# Syntax: np.linspace(start, stop, num_points)
# Unlike arange, STOP is INCLUSIVE

a = np.linspace(0, 1, 5)     # [0.   0.25  0.5   0.75  1.  ]
b = np.linspace(0, 10, 11)   # [0. 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.]
```

> 💡 Use `linspace` when you know **how many points** you want.  
> Use `arange` when you know the **step size**.

---

### 3.4 `np.zeros()`, `np.ones()`, `np.eye()`

```python
# All zeros — shape (3, 4)
z = np.zeros((3, 4))
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# All ones — shape (2, 3)
o = np.ones((2, 3))
# [[1. 1. 1.]
#  [1. 1. 1.]]

# Identity matrix — n x n with 1s on diagonal
e = np.eye(4)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]

# Full — fill with a specific value
f = np.full((3, 3), 7)
# [[7 7 7]
#  [7 7 7]
#  [7 7 7]]
```

---

### 3.5 Random Arrays

```python
# Uniform random floats between 0 and 1
r1 = np.random.rand(3, 4)         # shape (3, 4)

# Random integers — np.random.randint(low, high, size)
r2 = np.random.randint(1, 100, (3, 3))   # integers from 1 to 99

# Normal distribution (mean=0, std=1) — very common in ML!
r3 = np.random.randn(4, 4)

# Set seed for reproducibility (important in ML!)
np.random.seed(42)
r4 = np.random.rand(3)   # Always produces same output

# Random choice from an array
choices = np.random.choice([10, 20, 30, 40], size=5)  # picks 5 random values
```

---

## 4. Array Basics

### 4.1 Key Attributes

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(arr.ndim)    # 2        → number of dimensions
print(arr.shape)   # (2, 3)   → (rows, columns)
print(arr.size)    # 6        → total number of elements
print(arr.dtype)   # int64    → data type of elements
```

| Attribute | Meaning | Example Output |
|-----------|---------|---------------|
| `ndim` | Number of dimensions | `2` |
| `shape` | Tuple of dimension sizes | `(2, 3)` |
| `size` | Total elements | `6` |
| `dtype` | Element data type | `int64`, `float32` |

---

### 4.2 `astype()` — Change Data Type

```python
arr = np.array([1, 2, 3, 4])

# Convert integer array to float
arr_float = arr.astype(float)
print(arr_float)          # [1. 2. 3. 4.]
print(arr_float.dtype)    # float64

# Convert to int
arr_back = arr_float.astype(int)
print(arr_back)           # [1 2 3 4]

# Convert to bool (0 → False, non-zero → True)
arr_bool = arr.astype(bool)
print(arr_bool)           # [ True  True  True  True]
```

> 💡 In ML, you'll often need `float32` instead of `float64` to save memory when working with large datasets.

```python
arr = arr.astype(np.float32)
```

---

### 4.3 Converting Lists to Arrays

```python
# Simple list → 1D array
python_list = [10, 20, 30, 40]
arr = np.array(python_list)

# Nested list → 2D array
matrix_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix = np.array(matrix_list)
print(matrix.shape)   # (3, 3)
```

---

## 5. Indexing & Slicing ⚡ VERY IMPORTANT

> Mastering this section is critical. In ML, you'll be constantly extracting rows, columns, and subsets of your data.

### 5.1 1D Indexing & Slicing

```python
arr = np.array([10, 20, 30, 40, 50, 60])

# Indexing (0-based)
print(arr[0])     # 10   — first element
print(arr[-1])    # 60   — last element
print(arr[-2])    # 50   — second from last

# Slicing — arr[start:stop:step] (stop is EXCLUSIVE)
print(arr[1:4])   # [20 30 40]
print(arr[:3])    # [10 20 30]   — from beginning
print(arr[3:])    # [40 50 60]   — to end
print(arr[::2])   # [10 30 50]   — every 2nd element
print(arr[::-1])  # [60 50 40 30 20 10] — reversed
```

---

### 5.2 2D Indexing & Slicing

```python
matrix = np.array([[1,  2,  3,  4],
                   [5,  6,  7,  8],
                   [9, 10, 11, 12]])

# Single element — matrix[row, col]
print(matrix[0, 0])    # 1   — top-left
print(matrix[2, 3])    # 12  — bottom-right
print(matrix[-1, -1])  # 12  — same as above

# Entire row
print(matrix[1])        # [5 6 7 8]
print(matrix[1, :])     # [5 6 7 8]  — same

# Entire column
print(matrix[:, 2])     # [ 3  7 11]  — column at index 2

# Sub-matrix slicing — matrix[row_slice, col_slice]
print(matrix[0:2, 1:3])
# [[2 3]
#  [6 7]]

# First 2 rows, all columns
print(matrix[:2, :])
# [[1 2 3 4]
#  [5 6 7 8]]

# Last row, last 2 columns
print(matrix[-1, -2:])  # [11 12]
```

---

### 5.3 Boolean Indexing

```python
arr = np.array([10, 25, 5, 40, 15, 60])

# Create a boolean mask
mask = arr > 20
print(mask)          # [False  True False  True False  True]

# Apply mask — returns only elements where mask is True
print(arr[mask])     # [25 40 60]

# Can be done in one line
print(arr[arr > 20]) # [25 40 60]

# Multiple conditions (use & for AND, | for OR)
print(arr[(arr > 10) & (arr < 50)])  # [25 40 15]

# Use boolean indexing to modify values
arr[arr < 15] = 0
print(arr)   # [ 0 25  0 40 15 60]
```

> 💡 **This is extremely common in ML** — used to filter samples, select features, handle outliers, etc.

---

### 5.4 Fancy Indexing

```python
arr = np.array([10, 20, 30, 40, 50, 60])

# Pass a list of indices
indices = [0, 2, 5]
print(arr[indices])   # [10 30 60]

# 2D Fancy Indexing
matrix = np.array([[1,  2,  3],
                   [4,  5,  6],
                   [7,  8,  9]])

# Select specific rows
print(matrix[[0, 2]])
# [[1 2 3]
#  [7 8 9]]

# Select specific rows AND specific columns (pairs, not grid)
rows = [0, 1, 2]
cols = [0, 1, 2]
print(matrix[rows, cols])   # [1 5 9] — diagonal elements
```

---

## 6. Array Operations

### 6.1 Element-wise Arithmetic

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(a + b)    # [11 22 33 44]
print(b - a)    # [ 9 18 27 36]
print(a * b)    # [ 10  40  90 160]
print(b / a)    # [10. 10. 10. 10.]
print(a ** 2)   # [ 1  4  9 16]     — element-wise power
print(b % 3)    # [1 2 0 1]         — element-wise modulo
```

> ✅ All operations happen **element by element** — no loops needed!

---

### 6.2 Scalar Operations

```python
arr = np.array([1, 2, 3, 4, 5])

print(arr + 10)    # [11 12 13 14 15]
print(arr * 3)     # [ 3  6  9 12 15]
print(arr / 2)     # [0.5 1.  1.5 2.  2.5]
print(arr ** 2)    # [ 1  4  9 16 25]
```

---

### 6.3 Comparison Operations

```python
a = np.array([1, 5, 3, 8, 2])
b = np.array([2, 4, 3, 7, 9])

print(a > b)     # [False  True False  True False]
print(a == b)    # [False False  True False False]
print(a != b)    # [ True  True False  True  True]
print(a >= 3)    # [False  True  True  True False]
```

---

## 7. Broadcasting ⚡ CRITICAL

> Broadcasting is NumPy's most powerful (and most confusing) concept.  
> Once you truly understand this, you'll write cleaner and faster ML code.

### What is Broadcasting?

Broadcasting allows NumPy to perform operations on arrays of **different shapes** by automatically "stretching" the smaller array — **without copying data in memory**.

---

### Rule of Broadcasting

Two arrays are compatible for broadcasting if, for **each dimension** (compared from the right):
1. They are **equal**, OR
2. One of them is **1**

If neither condition is met → **Shape Mismatch Error**

---

### Example 1: Scalar + Array (simplest case)

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

result = arr + 10   # 10 is broadcast to every element
print(result)
# [[11 12 13]
#  [14 15 16]]
```

---

### Example 2: 1D Array + 2D Array (row broadcasting)

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])  # shape (3, 3)

row = np.array([10, 20, 30])    # shape (3,) → treated as (1, 3)

result = matrix + row           # row is broadcast across all 3 rows
print(result)
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]
```

```
Shapes:  (3, 3)
       + (   3)  → stretched to (3, 3)
```

---

### Example 3: Column vector broadcast

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])  # shape (3, 3)

col = np.array([[10],
                [20],
                [30]])          # shape (3, 1)

result = matrix + col           # col is broadcast across all 3 columns
print(result)
# [[11 12 13]
#  [24 25 26]
#  [37 38 39]]
```

```
Shapes:  (3, 3)
       + (3, 1)  → stretched to (3, 3)
```

---

### Example 4: Where Broadcasting FAILS

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)

b = np.array([1, 2])        # shape (2,)

# This will FAIL:
# result = a + b   → ValueError: operands could not be broadcast

# Fix: reshape b to a column vector
b_col = b.reshape(2, 1)     # shape (2, 1)
result = a + b_col           # Now works! shape (2, 3)
print(result)
# [[2 3 4]
#  [6 7 8]]
```

---

### Broadcasting Visual Summary

```
Scenario              Array A     Array B     Result
Scalar + Array        (4, 3)      ()          (4, 3) ✅
Row vector            (4, 3)      (3,)        (4, 3) ✅
Column vector         (4, 3)      (4, 1)      (4, 3) ✅
Incompatible          (4, 3)      (4,)        ERROR  ❌
```

---

## 8. Mathematical Functions

### 8.1 Core Statistical Functions

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(np.sum(arr))     # 45    — sum of ALL elements
print(np.mean(arr))    # 5.0   — mean of ALL elements
print(np.std(arr))     # 2.58  — standard deviation
print(np.var(arr))     # 6.67  — variance
print(np.min(arr))     # 1     — global minimum
print(np.max(arr))     # 9     — global maximum
print(np.median(arr))  # 5.0   — median
```

---

### 8.2 The `axis` Parameter ⚡ VERY IMPORTANT

> The `axis` parameter is one of the most confusing things for beginners.  
> Think of it like this: **axis=0 collapses rows, axis=1 collapses columns.**

```
Matrix:
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]

axis=0 → operate DOWN the rows (collapse rows → result has 1 row)
axis=1 → operate ACROSS the columns (collapse columns → result has 1 column)
```

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# axis=0: sum DOWN each column (result shape: (3,))
print(np.sum(arr, axis=0))    # [12 15 18]
#  1+4+7=12, 2+5+8=15, 3+6+9=18

# axis=1: sum ACROSS each row (result shape: (3,))
print(np.sum(arr, axis=1))    # [ 6 15 24]
#  1+2+3=6, 4+5+6=15, 7+8+9=24

# axis works for all functions
print(np.mean(arr, axis=0))   # [4. 5. 6.]   — mean of each column
print(np.max(arr, axis=1))    # [3 6 9]      — max of each row
print(np.min(arr, axis=0))    # [1 2 3]      — min of each column
```

**Memory tip:**
```
axis = 0 → "down" (between rows, vertically)
axis = 1 → "across" (between columns, horizontally)
```

---

### 8.3 Other Useful Math Functions

```python
arr = np.array([1, 4, 9, 16, 25])

print(np.sqrt(arr))    # [1. 2. 3. 4. 5.]   — square root
print(np.abs(arr))     # same (all positive)

arr2 = np.array([-3, -1, 0, 2, 5])
print(np.abs(arr2))    # [3 1 0 2 5]         — absolute value

# Exponential and log
print(np.exp([1, 2, 3]))      # [ 2.718  7.389  20.086]
print(np.log([1, np.e, 10]))  # [ 0.     1.     2.303]
print(np.log2([1, 2, 8]))     # [0. 1. 3.]
print(np.log10([1, 10, 100])) # [0. 1. 2.]

# Rounding
arr3 = np.array([1.23, 4.567, 8.9])
print(np.round(arr3, 1))   # [1.2 4.6 8.9]
print(np.floor(arr3))      # [1. 4. 8.]
print(np.ceil(arr3))       # [2. 5. 9.]

# Cumulative sum — very useful for probability distributions
arr4 = np.array([1, 2, 3, 4, 5])
print(np.cumsum(arr4))    # [ 1  3  6 10 15]
```

---

## 9. Reshaping & Transformations

### 9.1 `reshape()` — Change Shape Without Changing Data

```python
arr = np.arange(1, 13)     # [1 2 3 4 5 6 7 8 9 10 11 12]
print(arr.shape)            # (12,)

# Reshape to 3x4 matrix
matrix = arr.reshape(3, 4)
print(matrix)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

# Reshape to 4x3
matrix2 = arr.reshape(4, 3)

# Use -1 to let NumPy calculate one dimension automatically
matrix3 = arr.reshape(2, -1)    # NumPy auto-calculates: 12/2 = 6 → shape (2, 6)
matrix4 = arr.reshape(-1, 4)    # NumPy auto-calculates: 12/4 = 3 → shape (3, 4)
```

> ⚠️ Total elements must remain the same! `(12,)` can become `(3,4)` or `(2,6)` but NOT `(3,5)`.

---

### 9.2 `flatten()` vs `ravel()` — Collapse to 1D

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# flatten() — returns a COPY
flat_copy = matrix.flatten()
print(flat_copy)          # [1 2 3 4 5 6]

flat_copy[0] = 999        # modifying copy doesn't affect original
print(matrix[0, 0])       # 1 — original unchanged

# ravel() — returns a VIEW (faster, uses less memory)
flat_view = matrix.ravel()
flat_view[0] = 999        # modifying view DOES affect original!
print(matrix[0, 0])       # 999 — original changed!
```

> 💡 Use `flatten()` when you want safety (no side effects).  
> Use `ravel()` when performance matters.

---

### 9.3 `transpose()` — Flip Rows and Columns

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(matrix.shape)   # (2, 3)

# Transpose — rows become columns
T = matrix.T          # or matrix.transpose()
print(T)
# [[1 4]
#  [2 5]
#  [3 6]]
print(T.shape)        # (3, 2)
```

---

### 9.4 Adding/Removing Dimensions

```python
arr = np.array([1, 2, 3])   # shape (3,)

# Add a dimension — make it a row vector (1, 3)
row = arr[np.newaxis, :]      # or arr.reshape(1, -1)
print(row.shape)              # (1, 3)

# Add a dimension — make it a column vector (3, 1)
col = arr[:, np.newaxis]      # or arr.reshape(-1, 1)
print(col.shape)              # (3, 1)

# Squeeze — remove dimensions of size 1
squeezed = row.squeeze()
print(squeezed.shape)         # (3,)
```

---

## 10. Matrix Operations

### 10.1 Element-wise vs Matrix Multiplication

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Element-wise multiplication (NOT matrix multiplication)
print(A * B)
# [[ 5 12]
#  [21 32]]

# Matrix multiplication — np.dot() or @ operator
print(np.dot(A, B))
# [[19 22]
#  [43 50]]

print(A @ B)       # Same as np.dot(A, B) — preferred modern syntax
# [[19 22]
#  [43 50]]
```

> 💡 In ML, **`@` is the preferred operator** for matrix multiplication. It's cleaner and more readable.

---

### 10.2 How Matrix Multiplication Works

```
A (2×3)  @  B (3×4)  =  C (2×4)
     └── must match ──┘
```

```python
A = np.random.rand(2, 3)   # 2 rows, 3 cols
B = np.random.rand(3, 4)   # 3 rows, 4 cols
C = A @ B
print(C.shape)              # (2, 4) ✅

# This would FAIL:
# D = np.random.rand(3, 4)
# E = np.random.rand(3, 4)
# F = D @ E   → ERROR: (3,4) @ (3,4) — inner dims don't match
```

---

### 10.3 Other Matrix Functions

```python
A = np.array([[1, 2],
              [3, 4]])

# Determinant
print(np.linalg.det(A))          # -2.0

# Matrix inverse
print(np.linalg.inv(A))
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Eigenvalues and eigenvectors — used in PCA!
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)               # [-0.372  5.372]

# Norm (magnitude) of a vector
v = np.array([3, 4])
print(np.linalg.norm(v))         # 5.0  (Euclidean distance formula)
```

---

## 11. Useful Functions for ML

### 11.1 `np.unique()` — Find Unique Values

```python
labels = np.array([0, 1, 0, 2, 1, 2, 0, 3])

# Get unique values
print(np.unique(labels))           # [0 1 2 3]

# Get unique values AND their counts
values, counts = np.unique(labels, return_counts=True)
print(values)    # [0 1 2 3]
print(counts)    # [3 2 2 1]

# Very useful for checking class distribution in ML!
```

---

### 11.2 `np.where()` — Conditional Selection

```python
arr = np.array([10, -5, 20, -15, 30, -1])

# np.where(condition, value_if_true, value_if_false)
result = np.where(arr > 0, arr, 0)   # Replace negatives with 0
print(result)   # [10  0 20  0 30  0]

# Clip negatives to 0 (ReLU activation function in ML!)
relu = np.where(arr > 0, arr, 0)

# Get INDICES where condition is true
positive_indices = np.where(arr > 0)
print(positive_indices)   # (array([0, 2, 4]),)
print(arr[positive_indices])  # [10 20 30]
```

---

### 11.3 `np.sort()` and `np.argsort()`

```python
arr = np.array([40, 10, 50, 20, 30])

# Sort values (ascending by default)
print(np.sort(arr))         # [10 20 30 40 50]

# Sort descending
print(np.sort(arr)[::-1])   # [50 40 30 20 10]

# argsort() — returns INDICES that would sort the array
indices = np.argsort(arr)
print(indices)              # [1 3 4 0 2]  ← positions of 10, 20, 30, 40, 50

# Use argsort to sort another array by these positions
names = np.array(['d', 'a', 'e', 'b', 'c'])
print(names[np.argsort(arr)])   # ['a' 'b' 'c' 'd' 'e']  — names sorted by arr!
```

> 💡 `argsort()` is critical in ML — used for ranking predictions, finding nearest neighbors, etc.

---

### 11.4 Other Frequently Used ML Functions

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Stack arrays vertically (row-wise)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.vstack([a, b]))   # [[1 2 3], [4 5 6]]

# Stack arrays horizontally (column-wise)
print(np.hstack([a, b]))   # [1 2 3 4 5 6]

# Concatenate along an axis
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.concatenate([A, B], axis=0))   # Stack rows: shape (4, 2)
print(np.concatenate([A, B], axis=1))   # Stack cols: shape (2, 4)

# Clip — restrict values to a range
arr = np.array([-2, -1, 0, 1, 2, 3, 10])
print(np.clip(arr, 0, 5))   # [0 0 0 1 2 3 5]

# Argmax / Argmin — index of max/min (used in classification!)
predictions = np.array([0.1, 0.7, 0.2])
print(np.argmax(predictions))   # 1 → predicted class index
```

---

## 12. Performance Advantage

### Why NumPy is Faster Than Python Lists

NumPy achieves speed through:

1. **Contiguous Memory Layout** — Elements are stored next to each other in memory (cache-friendly)
2. **C Backend** — Operations run in compiled C code, not interpreted Python
3. **Vectorization** — Operations apply to entire arrays at once (no Python loops)
4. **BLAS/LAPACK** — Linear algebra uses hand-optimized CPU libraries

### Speed Comparison

```python
import numpy as np
import time

SIZE = 1_000_000

# Python List approach
py_list = list(range(SIZE))
start = time.time()
result = [x * 2 for x in py_list]
python_time = time.time() - start
print(f"Python list: {python_time:.4f}s")

# NumPy approach
np_arr = np.arange(SIZE)
start = time.time()
result = np_arr * 2
numpy_time = time.time() - start
print(f"NumPy array: {numpy_time:.4f}s")

print(f"NumPy is ~{python_time/numpy_time:.0f}x faster!")
# Typical output: NumPy is ~50-100x faster!
```

### Memory Comparison

```python
import sys
import numpy as np

SIZE = 1000

# Python list — each int is a Python object (28 bytes each)
py_list = list(range(SIZE))
py_memory = sys.getsizeof(py_list) + sum(sys.getsizeof(x) for x in py_list)
print(f"Python list: {py_memory} bytes")

# NumPy array — each int64 is just 8 bytes
np_arr = np.arange(SIZE)
np_memory = np_arr.nbytes
print(f"NumPy array: {np_memory} bytes")

print(f"NumPy uses {py_memory/np_memory:.1f}x less memory!")
# Typical: NumPy uses ~3.5x less memory
```

---

## 13. Common Mistakes

### ❌ Mistake 1: Shape Mismatch

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])    # shape (2, 3)

b = np.array([[1, 2],
              [3, 4],
              [5, 6]])       # shape (3, 2)

# WRONG — can't add arrays of different shapes
# a + b → ValueError!

# Matrix multiply IS valid because shapes (2,3) @ (3,2) = (2,2)
c = a @ b   # ✅ shape (2, 2)
```

---

### ❌ Mistake 2: Confusing `axis=0` and `axis=1`

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Which axis gives which result?
print(np.sum(arr, axis=0))   # [5 7 9]  — sum DOWN columns (3 results)
print(np.sum(arr, axis=1))   # [6 15]   — sum ACROSS rows (2 results)

# Memory trick: count the results
# axis=0 → result has as many values as there are COLUMNS (3)
# axis=1 → result has as many values as there are ROWS (2)
```

---

### ❌ Mistake 3: Copy vs View (Silent Bug!)

```python
original = np.array([1, 2, 3, 4, 5])

# Slicing returns a VIEW (not a copy!)
view = original[1:4]
view[0] = 999
print(original)    # [  1 999   3   4   5] ← ORIGINAL CHANGED!

# Fix: use .copy() to get an independent copy
copy = original[1:4].copy()
copy[0] = 0
print(original)    # [  1 999   3   4   5] ← original safe!

# Check if an array owns its data
print(view.base is original)   # True  → view
print(copy.base is None)       # True  → independent copy
```

> 🚨 **This is the most dangerous NumPy bug.** In ML pipelines, accidental in-place modification of training data causes hard-to-debug errors. Always use `.copy()` when you need independence.

---

### ❌ Mistake 4: Integer Division

```python
a = np.array([1, 2, 3])

# Python 3 / NumPy — division returns float
print(a / 2)      # [0.5 1.  1.5]  ✅

# But comparing dtypes matters
b = np.array([1, 2, 3], dtype=int)
c = np.array([1, 2, 3], dtype=float)

print((b / 2).dtype)   # float64 — OK
print((b + c).dtype)   # float64 — NumPy upcasts automatically
```

---

### ❌ Mistake 5: Not Using `np.random.seed()`

```python
# Without seed — results are different every time
print(np.random.rand(3))   # [0.37, 0.95, 0.73]  (random)
print(np.random.rand(3))   # [0.59, 0.15, 0.86]  (different!)

# With seed — reproducible results (critical for ML experiments!)
np.random.seed(42)
print(np.random.rand(3))   # [0.374, 0.951, 0.732]  (always same)
```

---

## 14. Mini Practice Section

> Try to solve these yourself before looking at the solutions. These cover real ML scenarios.

---

### Problem 1 — Create and Inspect

**Task:** Create a 4×5 matrix of random integers between 1 and 50. Print its shape, size, dtype, and min/max values.

```python
import numpy as np

np.random.seed(0)
matrix = np.random.randint(1, 51, (4, 5))

print("Array:\n", matrix)
print("Shape:", matrix.shape)       # (4, 5)
print("Size:", matrix.size)         # 20
print("Dtype:", matrix.dtype)       # int64
print("Min:", matrix.min())
print("Max:", matrix.max())
```

---

### Problem 2 — Slicing

**Task:** From the array below, extract the 2×2 submatrix from the center.

```python
arr = np.array([[ 1,  2,  3,  4,  5],
                [ 6,  7,  8,  9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]])

# Extract center 3x3
center = arr[1:4, 1:4]
print(center)
# [[ 7  8  9]
#  [12 13 14]
#  [17 18 19]]
```

---

### Problem 3 — Boolean Indexing

**Task:** Given an array of exam scores, replace all scores below 40 with 0 (failed), and print how many students failed.

```python
scores = np.array([85, 32, 67, 25, 90, 38, 72, 15, 60, 45])

failed_count = np.sum(scores < 40)
print(f"Students who failed: {failed_count}")   # 4

scores[scores < 40] = 0
print("Updated scores:", scores)
# [85  0 67  0 90  0 72  0 60 45]
```

---

### Problem 4 — Reshaping

**Task:** Create a 1D array from 1 to 24 and reshape it into (2, 3, 4) — a 3D array. Then extract the last 2D slice.

```python
arr = np.arange(1, 25).reshape(2, 3, 4)
print("Shape:", arr.shape)   # (2, 3, 4)

# Last 2D slice (index -1 along first axis)
print("Last slice:\n", arr[-1])
# [[13 14 15 16]
#  [17 18 19 20]
#  [21 22 23 24]]
```

---

### Problem 5 — Broadcasting

**Task:** Normalize a 3×4 matrix so that each column has values between 0 and 1 (min-max scaling — a core ML preprocessing step).

```python
np.random.seed(1)
data = np.random.randint(10, 100, (3, 4)).astype(float)
print("Original:\n", data)

col_min = data.min(axis=0)    # min of each column, shape (4,)
col_max = data.max(axis=0)    # max of each column, shape (4,)

normalized = (data - col_min) / (col_max - col_min)   # Broadcasting!
print("Normalized:\n", normalized.round(2))
# All values between 0.0 and 1.0 per column
```

---

### Problem 6 — Axis Operations

**Task:** Given a dataset of 5 students and 4 test scores, find each student's average score and which test was hardest (lowest average).

```python
scores = np.array([[85, 72, 90, 65],
                   [78, 85, 70, 88],
                   [92, 60, 75, 80],
                   [70, 88, 82, 75],
                   [65, 90, 78, 92]])

# Student averages (axis=1 → across columns)
student_avg = np.mean(scores, axis=1)
print("Student averages:", student_avg.round(1))
# [78.  80.2 76.8 78.8 81.2]

# Test averages (axis=0 → down rows)
test_avg = np.mean(scores, axis=0)
print("Test averages:", test_avg)
# [78.  79.  79.  80.]

# Hardest test (lowest average)
hardest_test = np.argmin(test_avg) + 1   # +1 for 1-based numbering
print(f"Hardest test: Test {hardest_test}")
```

---

### Problem 7 — Matrix Multiplication (Core ML Operation)

**Task:** Simulate a single forward pass of a tiny neural network layer: X (3 samples × 4 features) @ W (4 features × 2 neurons) + b (bias).

```python
np.random.seed(42)

X = np.random.rand(3, 4)       # 3 samples, 4 features
W = np.random.rand(4, 2)       # 4 inputs, 2 neurons
b = np.array([0.1, 0.2])       # 2 biases (one per neuron)

# Forward pass
output = X @ W + b             # Broadcasting: b is added to each row
print("Output shape:", output.shape)   # (3, 2)
print("Output:\n", output.round(3))
```

---

### Problem 8 — `where()` and `argsort()`

**Task:** Given prediction probabilities from a classifier, find the top-3 most confident predictions and their class indices.

```python
probabilities = np.array([0.05, 0.35, 0.12, 0.28, 0.08, 0.72, 0.15, 0.45])

# argsort gives ascending order; use [::-1] for descending
sorted_indices = np.argsort(probabilities)[::-1]
print("Indices sorted by confidence:", sorted_indices)
# [5 7 1 3 6 2 4 0]

# Top 3
top3_indices = sorted_indices[:3]
top3_probs   = probabilities[top3_indices]

print(f"Top-3 predictions: classes {top3_indices} with probs {top3_probs.round(2)}")
# Top-3: classes [5 7 1] with probs [0.72 0.45 0.35]
```

---

### Problem 9 — Copy vs View Safety

**Task:** You have training data. Create a validation set that you can modify independently without corrupting the original.

```python
training_data = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0],
                           [0.5, 1.5, 2.5]])

# WRONG approach — view shares memory
val_wrong = training_data[2:]
val_wrong[0, 0] = 999
print("Training data corrupted:", training_data[2, 0])  # 999!

# CORRECT approach — independent copy
training_data[2, 0] = 7.0     # restore
val_correct = training_data[2:].copy()
val_correct[0, 0] = 999
print("Training data safe:", training_data[2, 0])   # 7.0 ✅
```

---

### Problem 10 — End-to-End Mini Pipeline

**Task:** Simulate a basic data preprocessing pipeline: load data, normalize, split into features/labels, and compute class distribution.

```python
np.random.seed(99)

# Simulate a dataset: 100 samples, 3 features + 1 label column
data = np.random.rand(100, 3) * 100     # features 0-100
labels = np.random.randint(0, 3, 100)  # 3 classes: 0, 1, 2

# Step 1: Normalize features (z-score normalization)
mean = data.mean(axis=0)    # shape (3,)
std  = data.std(axis=0)     # shape (3,)
data_normalized = (data - mean) / std   # Broadcasting

# Step 2: Verify normalization
print("Mean after norm:", data_normalized.mean(axis=0).round(4))  # ~[0,0,0]
print("Std after norm:", data_normalized.std(axis=0).round(4))    # ~[1,1,1]

# Step 3: Class distribution
classes, counts = np.unique(labels, return_counts=True)
for c, n in zip(classes, counts):
    print(f"Class {c}: {n} samples ({n}%)")
```

---

## 15. Quick Revision Sheet

> 📋 Bookmark this section. Review it before every ML session.

---

### 🔷 Creating Arrays

```
np.array([...])             → from list
np.arange(start, stop, step) → like range()
np.linspace(start, stop, n) → n evenly spaced points
np.zeros((r, c))            → all zeros
np.ones((r, c))             → all ones
np.eye(n)                   → identity matrix
np.random.rand(r, c)        → uniform [0,1)
np.random.randn(r, c)       → normal dist (μ=0, σ=1)
np.random.randint(lo,hi,sz) → random integers
np.random.seed(n)           → reproducibility
```

---

### 🔷 Array Attributes

```
arr.ndim     → number of dimensions
arr.shape    → tuple (rows, cols, ...)
arr.size     → total elements
arr.dtype    → element data type
arr.astype() → convert dtype
```

---

### 🔷 Indexing & Slicing

```
arr[i]         → 1D: element at i
arr[i, j]      → 2D: row i, col j
arr[a:b]       → elements from a to b-1
arr[a:b, c:d]  → 2D sub-matrix
arr[::-1]      → reversed
arr[arr > 5]   → boolean indexing
arr[[0,2,4]]   → fancy indexing
```

---

### 🔷 Operations

```
arr + / - / * / **  → element-wise ops
arr + scalar        → scalar broadcast
arr1 + arr2         → element-wise (shapes must match/broadcast)
A @ B               → matrix multiplication
np.dot(A, B)        → same as @
```

---

### 🔷 Broadcasting Rules

```
Compare shapes from RIGHT to LEFT:
- Equal dims     → compatible ✅
- One dim is 1  → stretched ✅
- Neither above → ERROR ❌

Common patterns:
(4, 3) + (3,)   → (4, 3) ✅  row broadcast
(4, 3) + (4,1)  → (4, 3) ✅  column broadcast
(4, 3) + (4,)   → ERROR  ❌  use reshape!
```

---

### 🔷 Math Functions

```
np.sum(arr, axis=?)   → sum (axis=0: cols, axis=1: rows)
np.mean(arr, axis=?)  → mean
np.std(arr, axis=?)   → standard deviation
np.min / np.max       → min/max
np.argmin / np.argmax → index of min/max
np.sqrt(arr)          → square root
np.abs(arr)           → absolute value
np.exp(arr)           → e^x
np.log(arr)           → natural log
np.cumsum(arr)        → cumulative sum
```

---

### 🔷 Reshaping

```
arr.reshape(r, c)   → new shape (same data)
arr.reshape(-1, c)  → auto-calculate rows
arr.flatten()       → 1D copy
arr.ravel()         → 1D view
arr.T               → transpose
arr[:, np.newaxis]  → add column dimension
arr[np.newaxis, :]  → add row dimension
arr.squeeze()       → remove size-1 dims
```

---

### 🔷 ML Utility Functions

```
np.unique(arr, return_counts=True)  → unique values + counts
np.where(cond, val_true, val_false) → conditional replacement
np.argsort(arr)                     → sort indices (ascending)
np.sort(arr)[::-1]                  → sort descending
np.clip(arr, lo, hi)                → restrict values
np.concatenate([a,b], axis=?)       → join arrays
np.vstack([a,b])                    → stack rows
np.hstack([a,b])                    → stack columns
```

---

### 🔷 Critical Gotchas

```
⚠️  Slicing returns a VIEW — use .copy() for independence
⚠️  axis=0 → down rows (collapse rows)
⚠️  axis=1 → across columns (collapse columns)
⚠️  Matrix multiply: inner dims must match → (m,k) @ (k,n) = (m,n)
⚠️  Always set np.random.seed() for reproducibility
⚠️  Use .astype(np.float32) for memory efficiency in ML
```

---

### 🔷 ML Preprocessing Cheat Sheet

```python
# Z-score normalization (standardization)
normalized = (X - X.mean(axis=0)) / X.std(axis=0)

# Min-Max scaling (normalization)
scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# One-hot encoding alternative
labels = np.array([0, 1, 2, 1, 0])
one_hot = np.eye(3)[labels]

# Train/test split
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]

# Dot product for predictions (linear model)
predictions = X @ weights + bias

# ReLU activation (used in neural networks)
relu = np.where(x > 0, x, 0)   # or: np.maximum(0, x)

# Softmax activation
def softmax(x):
    e_x = np.exp(x - np.max(x))   # subtract max for numerical stability
    return e_x / e_x.sum()
```

---

> ✅ **You are now ready to start Machine Learning with NumPy.**  
> The next steps are: **Pandas → Matplotlib → Scikit-learn**  
> Every concept you learned here will appear — daily — in your ML journey.

---

*Notes compiled for ML beginners. Covers NumPy from scratch to intermediate level.*  
*All examples are self-contained and runnable with `import numpy as np`.*
