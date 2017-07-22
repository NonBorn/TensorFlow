import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
print A
# [[1 2 3]
# [4 5 6]]
Af = np.array([1, 2, 3], float)
print Af

####################################################################

A = np.arange(0, 1, 0.2)
print A

A = np.linspace(0, 2*np.pi, 4)
print A

A = np.zeros([2,3])
print A
# np.ones, np.diag

A.shape
####################################################################

A = np.random.random((2,3))
print A

a = np.random.normal(loc=1.0, scale=2.0, size=(2,2))
print a

np.savetxt("a_out.txt", a)
a.tofile('foo.csv',sep=',',format='%10.5f')
np.savetxt("foo.csv", a, delimiter=",")

b = np.loadtxt("a_out.txt")
print b

b = np.loadtxt("foo.csv", delimiter=',')
print b

####################################################################

A = np.eye(4, k=0, dtype=float)
print A

A = np.eye(4, k=1, dtype=float)
print A

A = np.eye(4, k=-1, dtype=float)
print A

####################################################################

import numpy as np
u = [1, 2, 3]
v = [1, 1, 1]

print np.unique(v)

print np.inner(u, v)

print np.outer(u, v)

print np.dot(u, v)

# see https://www.tutorialspoint.com/numpy/numpy_tutorial.pdf

####################################################################

a = np.array([1,2,3], float)
b = np.array([5,2,6], float)

print a + b
print a - b
print a * b
print b / a
print a % b
print b**a

####################################################################

a = np.array([1.1,2.7,3.5], float)
print np.floor(a)
print np.ceil(a)
print np.rint(a)
print a.sum()
print a.prod()
print a.mean()
print a.var()
print a.std()
print a.min()
print a.max()

print a.clip(2.5, 3)

####################################################################

a = np.array([[1,2], [3,4]], float)
b = np.array([[2,0], [1,3]], float)
print a
print b
print a * b
print a[1:2,:]

print np.dot(a, b)

####################################################################



####################################################################