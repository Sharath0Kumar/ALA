from vec import Vec
import time 
import timeit

sizes= [2000, 4000, 8000, 16000, 32000, 64000]

# for size in sizes:
#     v1 = Vec.uniform(size)
#     v2 = Vec.uniform(size)
    
    # start = time.perf_counter()
    # result = v1 + v2
    # end = time.perf_counter()
    # print("Time : ", end -start ,"seconds")

    # time_taken = timeit.timeit(lambda: v1 + v2, number=10)
    # average_time = time_taken / 10
    # print(f"Size: {size}, Average Time: {average_time:.8f} seconds")


#Better way 

def measure_time(operation , number = 10):
    total_time = timeit.timeit(operation, number=number)
    return total_time / number

for size in sizes:
    v1 = Vec.uniform(size)
    v2 = Vec.uniform(size)

    addition_time = measure_time(lambda: v1 + v2)
    subtraction_time = measure_time(lambda: v1 - v2)
    multiplication_time = measure_time(lambda: 2 * v1)
    inplace_multiplication_time = measure_time(lambda: v1.__imul__(2))  
    inplace_addition_time = measure_time(lambda: v1.__iadd__(v2))
    negation_time = measure_time(lambda: -v1)
    norm_time = measure_time(lambda: v1.norm())

    print(f"-" *10 + f"Size : {size}" + "-"*10)

    print(f"Size: {size}, Addition Time: {addition_time:.8f} seconds")
    print(f"Size: {size}, Subtraction Time: {subtraction_time:.8f} seconds")
    print(f"Size: {size}, Multiplication Time: {multiplication_time:.8f} seconds")
    print(f"Size: {size}, In-place Multiplication Time: {inplace_multiplication_time:.8f} seconds")
    print(f"Size: {size}, In-place Addition Time: {inplace_addition_time:.8f} seconds")
    print(f"Size: {size}, Negation Time: {negation_time:.8f} seconds")  
    print(f"Size: {size}, Norm Time: {norm_time:.8f} seconds")  

    

