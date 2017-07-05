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

    for i in range(len(a[:, 0])):
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

    min_rank = min((n, m))
    for i, j in zip(range(min_rank - 1), range(min_rank - 1)):
        half_sum = (a[i + 1, j] + a[i, j + 1]) / 2
        a[i + 1, j], a[i, j + 1] = half_sum, half_sum

    print(a)

def t2_26():
    n, m = map(int, input('enter n and m: ').split())
    ar = reduce(lambda res, x: res + [random.randrange(-50, 101)], range(n * m), [])
    a = numpy.array(ar).reshape(n, m)

    print(a)

    l, k = map(int, input('enter l and k: ').split())

    split = numpy.hsplit(a, (0, l))
    first_and_second = numpy.vsplit(split[1], (0, k))
    list.__delitem__(first_and_second, 0)
    third_and_fourth = numpy.vsplit(split[2], (0, k))
    list.__delitem__(third_and_fourth, 0)

    print(*first_and_second[0], sep = '\n', end = '\n')
    print('average in first part: ',
          sum(sum(first_and_second[0])) /
          (first_and_second[0].shape[0] * first_and_second[0].shape[1]))

    print(*first_and_second[1], sep = '\n', end = '\n')
    print('average in second part: ',
          sum(sum(first_and_second[1])) /
          (first_and_second[1].shape[0] * first_and_second[1].shape[1]))

    print(*third_and_fourth[0], sep = '\n', end = '\n')
    print('average in third part: ',
          sum(sum(third_and_fourth[0])) /
          (third_and_fourth[0].shape[0] * third_and_fourth[0].shape[1]))

    print(*third_and_fourth[1], sep = '\n', end = '\n')
    print('average in fourth part: ',
          sum(sum(third_and_fourth[1])) /
          (third_and_fourth[1].shape[0] * third_and_fourth[1].shape[1]))

