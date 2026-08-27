from vec import Vec
import numpy as np
import timeit

sizes = [2000, 4000, 8000, 16000, 32000, 64000]

REPEATS = 10

def measure(operation):
    total = timeit.timeit(operation , number = REPEATS)
    return total / REPEATS

for size in sizes:
    v1 = Vec.uniform(size)
    v2 = Vec.uniform(size)

    a = np.random.uniform(0,1,size)
    b = np.random.uniform(0,1,size)

    vec_add = measure(lambda : v1 + v2)
    numpy_add = measure(lambda : a + b)

    vec_sub = measure(lambda : v1 - v2)
    numpy_sub = measure(lambda : a - b)

    vec_mul = measure(lambda : 2 *v1)
    numpy_mul = measure(lambda : 2 * a)

    vec_neg = measure(lambda : -v1)
    numpy_neg = measure(lambda : -a)

    vec_norm = measure(lambda : v1.norm())
    numpy_norm = measure(lambda : np.linalg.norm(a))

    print(f"-"*10 + f"Size : {size}" + "-"*10)

    print(f"Vec Add  Time: {vec_add:.8f}  seconds \t Numpy Add  Time: {numpy_add:.8f}  seconds")
    print(f"Vec Sub  Time: {vec_sub:.8f}  seconds \t Numpy Sub  Time: {numpy_sub:.8f}  seconds")
    print(f"Vec Mul  Time: {vec_mul:.8f}  seconds \t Numpy Mul  Time: {numpy_mul:.8f}  seconds")
    print(f"Vec Neg  Time: {vec_neg:.8f}  seconds \t Numpy Neg  Time: {numpy_neg:.8f}  seconds")
    print(f"Vec Norm Time: {vec_norm:.8f}  seconds \t Numpy Norm Time: {numpy_norm:.8f}  seconds")