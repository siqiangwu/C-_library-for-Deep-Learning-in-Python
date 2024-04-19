import numpy as np
import easynn as nn

# Create a numpy array of 10 rows and 5 columns.
# Set the element at row i and column j to be i+j.
def Q1():
    rows = 10
    columns = 5
    array = np.empty((rows,columns))
    for i in range (rows):
        for j in range (columns):
            array[i,j] = i+j  
    return array

# Add two numpy arrays together.
def Q2(a, b):
    return a+b

# Multiply two 2D numpy arrays using matrix multiplication.
def Q3(a, b):
    return np.matmul(a,b)

# For each row of a 2D numpy array, find the column index
# with the maximum element. Return all these column indices.
def Q4(a):
    return np.argmax(a, axis=1)

# Solve Ax = b.
def Q5(A, b):
    return np.linalg.solve(A,b)

# Return an EasyNN expression for a+b.
def Q6():
    a = nn.Input("a") 
    b = nn.Input("b")
    c = a+b
    return c

# Return an EasyNN expression for a+b*c.
def Q7():
    a = nn.Input("a") 
    b = nn.Input("b")
    c = nn.Input("c")
    d = a+b*c
    return d

# Given A and b, return an EasyNN expression for Ax+b.
def Q8(A, b):
    a1 = nn.Const(A)
    b1 = nn.Const(b)
    x = nn.Input("x")
    y = a1*x+b1
    return y

# Given n, return an EasyNN expression for x**n.
def Q9(n):
    x = nn.Input("x")
    y = x
    for _ in range(int(n)-1):
        y = y*x
    return y

# Return an EasyNN expression to compute
# the element-wise absolute value |x|.
def Q10():
    x = nn.Input("x")
    relu = nn.ReLU()
    d = relu(x) + relu(-x)
    return d
