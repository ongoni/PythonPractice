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
    ar = reduce(lambda res, x: res + [random.randrange(-50, 101)], range(n * m), [])
    a = numpy.array(ar).reshape(n, m)

    print(a)

    for i in range(n):
        a[i] += a[l]

    print(a)

def t2_16():
    n, m = map(int, input('enter n and m: ').split())
    ar = reduce(lambda res, x: res + [random.randrange(-50, 101)], range(n * m), [])
    a = numpy.array(ar).reshape(n, m)

    print(a, end = '\n\n')

    l = int(input('enter l: '))

    print(numpy.delete(a, l, axis = 0))

def t2_21():
    n, m = map(int, input('enter n and m: ').split())
    ar = reduce(lambda res, x: res + [random.randrange(-50, 101)], range(n * m), [])
    a = numpy.array(ar).reshape(n, m)

    print(a)

    min_dim = min((n, m))
    for i in range(min_dim - 1):
        half_sum = (a[i + 1, i] + a[i, i + 1]) / 2
        a[i + 1, i] = a[i, i + 1] = half_sum

    print(a)

def t2_26():
    n, m = map(int, input('enter n and m: ').split())
    ar = reduce(lambda res, x: res + [random.randrange(-50, 101)], range(n * m), [])
    a = numpy.array(ar).reshape(n, m)

    print(a)

    l, k = map(int, input('enter l and k: ').split())

    first = a[:l, :k]
    print(first, sep = '\n', end = '\n')
    print('average in first part: ',
          sum(sum(first)) /
          (first.shape[0] * first.shape[1]))

    second = a[:l, k:]
    print(second, sep = '\n', end = '\n')
    print('average in second part: ',
          sum(sum(second)) /
          (second.shape[0] * second.shape[1]))

    third = a[l:, :k]
    print(third, sep = '\n', end = '\n')
    print('average in third part: ',
          sum(sum(third)) /
          (third.shape[0] * third.shape[1]))

    fourth = a[l:, k:]
    print(fourth, sep = '\n', end = '\n')
    print('average in fourth part: ',
          sum(sum(fourth)) /
          (fourth.shape[0] * fourth.shape[1]))

