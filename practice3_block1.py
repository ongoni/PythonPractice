import numpy
import random
from functools import reduce

def t1_1():
    def get_element_w_zero_before(vec):
        prev_is_zero = False
        result = []
        for i in range(len(vec)):
            if prev_is_zero:
                result.append(vec[i])
                prev_is_zero = False
                continue
            if vec[i] == 0:
                prev_is_zero = True
                continue
        return result

    def t1(a):
        print('1. 4x4 matrix:')
        print(a)

    def t2(a):
        print('2. element with indexes [2, 3]: ', a[2, 3])

    def t3(a):
        print('3. first matrix line: ', a[0])

    def t4(a):
        print('4. every second element in 3rd line: ', a[2, ::2])

    def t5(a):
        c = a.reshape(8, 2)
        print('5. 8x2 matrix: ', c)
        return c

    def t6(c):
        print('6. matrix multipying by scalar: ')
        scalar = int(input('enter scalar: '))
        c = c.dot(scalar)
        print(c)
        return c

    def t7(c):
        print('7: minimum in every line: ', [min(c[i]) for i in range(c.shape[0])])

    def t8(c):
        print('8. maximum in last column: ', max(c[:, c.shape[1] - 1]))

    def t9(a):
        v = numpy.array(a.reshape(1, a.shape[0] * a.shape[1]))[0].tolist()
        print('9. maximum in elements which have 0 before it: ', max(get_element_w_zero_before(v)))

    def t10():
        result = 1
        for i in range(a.shape[0]):
            result *= a[i, i]

        print('10. result of multiplying elements from main giagonal: ', result)

    ar = reduce(lambda res, x: res + [random.randrange(-50, 101)], range(4 * 4), [])
    a = numpy.array(ar).reshape(4, 4)

    t1(a)
    t2(a)
    t3(a)
    t4(a)
    c = t5(a)
    c = t6(c)
    t7(c)
    t8(c)
    t9(a)
    t10()
