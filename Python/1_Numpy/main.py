import numpy as np
import time  # For sleep in iteration examples

# -----------------------------
# Python list vs NumPy array
# -----------------------------
# Multiplying a Python list repeats the list
my_list = [1, 2, 3] * 2
print(type(my_list), my_list)  # <class 'list'> [1, 2, 3, 1, 2, 3]

# Multiplying a NumPy array performs element-wise multiplication
arr = np.array([1, 2, 3]) * 2
print(type(arr), arr)  # <class 'numpy.ndarray'> [2 4 6]

# -----------------------------
# Array dimensions
# -----------------------------
# 0-D array (scalar)
arr_0d = np.array(42)
print(arr_0d.ndim, arr_0d)  # 0-D

# 1-D array
arr_1d = np.array([1, 2, 3, 4, 5])
print(arr_1d.ndim, arr_1d)  # 1-D

# 2-D array (matrix)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d.ndim, arr_2d)  # 2-D

# 3-D array (tensor)
arr_3d = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])
print(arr_3d.ndim, arr_3d)  # 3-D

# Higher dimensional array (using ndmin)
arr_6d = np.array([1, 2, 4, 6], ndmin=6)
print(arr_6d)

# -----------------------------
# Array indexing and slicing
# -----------------------------
arr = np.array([2, 5, 6, 8, 9])
print(arr[2])           # Access single element: 6
print(arr[1] + arr[3])  # Sum of two elements: 13

# Slicing arrays
print(arr[:5])   # From start to index 4
print(arr[2:])   # From index 2 to end
print(arr[1:4])  # From index 1 to 3
print(arr[0:6:2]) # From index 0 to 5, step 2

# 2-D indexing
arr_2d = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr_2d[0,0])  # Row 0, Column 0
print(arr_2d[1,0])  # Row 1, Column 0

# 3-D indexing
arr_3d = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])
print(arr_3d[1,1,2])  # Access 12

# -----------------------------
# Array creation functions
# -----------------------------
print(np.arange(10))             # Array from 0 to 9
print(np.zeros((3,3)))           # 3x3 zero matrix
print(np.ones((4,4)))            # 4x4 matrix of ones
print(np.full((4,4), 5))         # 4x4 matrix filled with 5
print(np.eye(4))                 # 4x4 identity matrix
print(np.linspace(0,1,5))        # 5 evenly spaced values from 0 to 1
print(np.random.rand(2,3))       # 2x3 matrix with random floats [0,1)
print(np.random.randint(0,10,5)) # 5 random integers from 0 to 9
print(np.empty((2,4)))           # 2x4 empty matrix (uninitialized)

# -----------------------------
# Scalar & vectorized arithmetic
# -----------------------------
arr = np.array([1,2,3])
print(arr+1, arr-1, arr*2, arr/3, arr**3)  # Element-wise arithmetic

# Arithmetic between arrays
arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
print(np.add(arr1, arr2))      # Element-wise addition
print(arr1 - arr2)              # Subtraction
print(np.multiply(arr1,2))      # Multiply by scalar
print(np.divide(arr1, arr2))    # Element-wise division
print(np.pow(arr1,2))           # Square each element
print(np.mod(arr1, arr2))       # Modulus

# -----------------------------
# Comparison and filtering
# -----------------------------
scores = np.array([95,30,35,90,19,52,84,10])
print(scores <50, scores >50, scores ==90)  # Boolean comparisons
scores[scores < 60] = 0                      # Replace failing scores with 0
print(scores)

# Advanced filtering using conditions
age = np.array([[17,10,21,8,16,90,55,35],[39,46,13,99,55,64,75,79]])
print(age[age < 18])                  # Minors
print(age[(age >= 18) & (age < 55)])  # Adults
print(age[age >= 60])                 # Seniors

# -----------------------------
# Array iteration
# -----------------------------
arr = np.array([2,4,6,8,10])
for i in arr:
    time.sleep(0.05)  # Pause for demonstration
    print(i)

# Iterating 2-D array element-wise
arr2d = np.array([[2,4,6,8,10],[12,14,16,18,20]])
for row in arr2d:
    for elem in row:
        print(elem)

# -----------------------------
# Copy vs View
# -----------------------------
arr = np.array([1,2,3,4,5])
x_view = arr.view()  # Creates a view (changes in original reflect here)
arr[0] = 42
print(x_view)

x_copy = arr.copy()  # Creates a copy (independent of original)
arr[0] = 1
print(x_copy)

# -----------------------------
# Set operations
# -----------------------------
arr = np.array([1,2,1,3,4,5,4,5,4,6,7,8,7,4,7,8])
print(np.unique(arr))  # Unique elements

arr1 = np.array([1,2,3,4,5])
arr2 = np.array([3,4,5,6])
print(np.union1d(arr1, arr2))       # Union
print(np.intersect1d(arr1, arr2))   # Intersection
print(np.setdiff1d(arr1, arr2))     # Difference (arr1 - arr2)
print(np.setxor1d(arr1, arr2))      # Symmetric difference

# -----------------------------
# Sorting
# -----------------------------
arr = np.array([2,8,0,6,2,9,9,3])
print(np.sort(arr))  # Numeric sort

arr_str = np.array(['orange','apple','mango','banana','lemon'])
print(np.sort(arr_str))  # Alphabetical sort

# -----------------------------
# Concatenation & splitting
# -----------------------------
arr1 = np.array([1,2,5,10])
arr2 = np.array([11,12,15,20])
print(np.concatenate((arr1, arr2)))      # Concatenate arrays

arr = np.array([0,1,2,3,4,5,6,7,8,9])
print(np.array_split(arr, 2))            # Split array into 2 parts

# -----------------------------
# Reshaping, flattening, stacking
# -----------------------------
arr = np.arange(12)
reshaped = arr.reshape((3,4))  # 3x4 matrix
flattened = reshaped.flatten()  # Flatten back to 1-D
print(reshaped)
print(flattened)

# Vertical and horizontal stacking
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
print(np.vstack((arr1, arr2)))  # Stack vertically
print(np.hstack((arr1, arr2)))  # Stack horizontally

# Broadcasting example
arr = np.array([[1,2,3],[4,5,6]])
print(arr + np.array([1,1,1]))  # Adds [1,1,1] to each row



# -----------------------------
# Maths - Stats , Calculus, probability
# -----------------------------

# -----------------------------
# Statistics in NumPy
# -----------------------------

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print("Array:", arr)

# Measures of central tendency
print("Mean:", np.mean(arr))          # Average value
print("Median:", np.median(arr))      # Middle value
print("Weighted Mean:", np.average(arr, weights=[1,2,1,2,1,2,1,2,1,2]))  # Weighted mean

# Measures of dispersion
print("Standard Deviation:", np.std(arr))  # Spread of data
print("Variance:", np.var(arr))            # Square of standard deviation
print("Min:", np.min(arr))
print("Max:", np.max(arr))
print("Range:", np.ptp(arr))               # Peak-to-peak (max-min)

# Percentiles & Quantiles
print("25th Percentile:", np.percentile(arr, 25))
print("50th Percentile (median):", np.percentile(arr, 50))
print("75th Percentile:", np.percentile(arr, 75))
print("Quartiles (np.quantile):", np.quantile(arr, [0.25,0.5,0.75]))


# -----------------------------
# Probability & Random Numbers
# -----------------------------

# Random numbers
print("Random float [0,1):", np.random.rand())
print("Random integer [0,10):", np.random.randint(0,10))
print("Random normal distribution (mean=0,std=1):", np.random.randn(5))  

# Probability distribution functions using simulation
n = 100000
# Coin toss simulation (0=heads, 1=tails)
coin = np.random.randint(0,2,n)
print("Probability of heads:", np.mean(coin==0))
print("Probability of tails:", np.mean(coin==1))

# Dice roll simulation
dice = np.random.randint(1,7,n)
print("Probability of rolling a 6:", np.mean(dice==6))

# Bernoulli / Binomial
print("Binomial (10 trials, p=0.5):", np.random.binomial(n=10, p=0.5, size=5))

# Uniform distribution
print("Uniform [0,1]:", np.random.uniform(0,1,5))

# -----------------------------
# Calculus-related operations
# -----------------------------

# 1. Derivative (finite differences)
x = np.linspace(0, 10, 100)
y = x**2  # y = f(x)
dy_dx = np.gradient(y, x)  # Numerical derivative
print("dy/dx at x[50]:", dy_dx[50])

# 2. Integration (numerical using trapezoid rule)
area = np.trapz(y, x)  # Approximate integral of y w.r.t x
print("Integral of y = x^2 from 0 to 10:", area)

# 3. Cumulative sum (analogous to discrete integral)
cum_y = np.cumsum(y) * (x[1]-x[0])
print("Cumulative sum (discrete integral) at last element:", cum_y[-1])

# 4. Vectorized calculus functions
f = np.sin(x)  # example function
df_dx = np.gradient(f, x)
integral_f = np.trapz(f, x)
print("Derivative of sin(x) at x[50]:", df_dx[50])
print("Integral of sin(x):", integral_f)

# -----------------------------
# Correlation & Covariance
# -----------------------------

# Sample data
x = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])

print("Covariance matrix:\n", np.cov(x,y))
print("Correlation coefficient:\n", np.corrcoef(x,y))



# -----------------------------
# Saving and loading arrays
# -----------------------------
np.save("array.npy", arr)     # Save array to file
loaded_arr = np.load("array.npy")  # Load array from file
print(loaded_arr)

