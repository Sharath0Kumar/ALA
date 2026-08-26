from vec import Vec

# For vec.zeros()
v =  Vec.zeros(5)
assert v.elements == [0,0,0,0,0]
print("Vec.zeros() passed")

# for vec.add
v1 = Vec([1,3,5])
v2 = Vec([2,4,4])

result = v1 + v2
assert result.elements == [3,7,9]
print("Vec.__add__() passed")

result1 = v1 - v2
assert result.elements == [-1,-1,1] 
print("Vec.__sub__() passed")

v3 = Vec([1,2,3])
result1 = 2*v3
assert result.elements == [2,4,6]
print("Vec.__rmul__() passed")

v4 = Vec([1,2,3])
v4 *= 3
assert v.elements == [3,6,9]
print("Vec.__imul__() passed")

v5 = Vec([1,2,3])
result = -v
assert result.elements == [-1,-2,-3]
print("Vec.__neg__() passed")

v1 = Vec([1,2,3])
v2 = Vec([4,5,6])
v1 += v2
assert v1.elements == [5,7,9]
print("Vec.__iadd__() passed")

v6 = Vec.ones(4)
assert v.elements == [1, 1, 1, 1]
print("ones() test passed")

v7 = Vec([10, 20, 30, 40])
assert len(v) == 4
print("len() test passed")

v = Vec([3, 4])
assert v.norm() == 5
print("norm test passed")

try :
    Vec([1,3,"Helloji"])
    assert False
except TypeError:
    print("TypeError test passed")