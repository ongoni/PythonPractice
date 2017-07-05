import numpy
import random
from functools import reduce

def t2_1():
    n, m = map(int, input('enter n and m: ').split())
    ar = reduce(lambda res, x: res + [random.randrange(-50, 101)], range(n * m), [])
    a = numpy.array(ar).reshape(n, m)
    print(a)

    m, index = a[0, 0], 0
    for i in range(a.shape[1]):
        t = sum(a[:, i])
        if m < t:
            m = t
            index = i
    print(max(a[:, index]))

def t2_6():
    n, m = map(int, input('enter n and m: ').split())
    ar = reduce(lambda res, x: res + [random.randrange(-50, 101) + random.random()], range(n * m), [])
    a = numpy.array(ar).reshape(n, m)
    print(a, end = '\n\n')
    sum_of_columns, sum_of_all = sum(a), sum(sum(a))
    b = [numpy.array([e / (sum_of_all / 100) for e in sum_of_columns])]
    a = numpy.concatenate((a, b), 0)
    print(a)

def t2_11():
    n, m, l = map(int, input('enter n, m and l: ').split())
    ar = reduce(lambda res, x: res + [random.randrange(-50, 101) + random.random()], range(n * m), [])
    a = numpy.array(ar).reshape(n, m)

