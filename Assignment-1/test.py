from vec import Vec
import math


# Zeros test
zero_vector = Vec.zeros(5)
assert zero_vector.elements == (0, 0, 0, 0, 0)
print("zeros test successful")


# Addition test
v1 = Vec([1, 3, 5])
v2 = Vec([2, 4, 4])

addition = v1 + v2
assert addition.elements == (3, 7, 9)
print("addition test successful")


# Subtraction test
subtraction = v1 - v2
assert subtraction.elements == (-1, -1, 1)
print("subtraction test successful")


# Scalar multiplication test
v3 = Vec([1, 2, 3])

multiplication = 2 * v3
assert multiplication.elements == (2, 4, 6)
print("scalar multiplication test successful")


# In-place multiplication test
v4 = Vec([1, 2, 3])
v4 *= 3
assert v4.elements == (3, 6, 9)
print("in-place multiplication test successful")


# Negation test
v5 = Vec([1, 2, 3])
negative = -v5
assert negative.elements == (-1, -2, -3)
print("negation test successful")


# In-place addition test
v6 = Vec([1, 2, 3])
v7 = Vec([4, 5, 6])
v6 += v7
assert v6.elements == (5, 7, 9)
print("in-place addition test successful")


# Ones test
one_vector = Vec.ones(4)
assert one_vector.elements == (1, 1, 1, 1)
print("ones test successful")


# Length test
v8 = Vec([10, 20, 30, 40])
assert len(v8) == 4
print("length test successful")


# Norm test
v9 = Vec([3, 4])
assert v9.norm() == 5
print("norm test successful")


# Invalid element test
try:
    Vec([1, 3, "Hellohiiiiholaaaa"])
    assert False
except TypeError:
    print("invalid element test successful")


# Dimension mismatch test
try:
    Vec([1, 2]) + Vec([1, 2, 3])
    assert False
except TypeError:
    print("dimension mismatch test successful")


# Assignment 1: Mean
v10 = Vec([2, 4, 6, 8])
assert v10.mean() == 5
print("mean test successful")


# Assignment 1: Demean
v11 = Vec([2, 4, 6])
demeaned = v11.demean()

assert demeaned.elements == (-2.0, 0.0, 2.0)
print("demean test successful")


# Assignment 1: Mean of demeaned vector should be zero
v12 = Vec([5, 10, 15, 20])
demeaned_v12 = v12.demean()

assert math.isclose(demeaned_v12.mean(), 0.0, abs_tol=1e-9)
print("demeaned vector mean test successful")


# Assignment 1: Standard deviation
v13 = Vec([1, 3, 5, 7])

assert math.isclose(v13.std(), math.sqrt(5), rel_tol=1e-9)
print("standard deviation test successful")


# Assignment 1: Standard deviation of equal values should be zero
v14 = Vec([9, 9, 9, 9])

assert math.isclose(v14.std(), 0.0, abs_tol=1e-9)
print("zero standard deviation test successful")


print("All tests passed successfully")