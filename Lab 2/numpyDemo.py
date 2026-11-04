import numpy as np
a = np.array([1,2,3])

# assert (a.dtype == np.int64)

assert(a.shape == (3,))

assert (isinstance(a.shape , tuple))


b = np.zeros(4)
assert b == [0,0,0,0]

assert np.array_equal(b, [0,0,0,0])

assert (a== np.array([0,0,0,0])).all()