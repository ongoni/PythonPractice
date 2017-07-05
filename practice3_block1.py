import numpy

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

    # a = numpy.ndarray((4, 4), int)
    # a.fill(0)

    b = numpy.array([
        [1, 0, 3, 4],
        [5, 6, 0, 8],
        [9, 0, 1, 2],
        [3, 4, 0, 6]
    ])

    print('1. 4x4 matrix:')
    print(b)

    print('2. element with indexes [2, 3]: ', b[2, 3])

    print('3. first matrix line: ', b[0])

    print('4. every second element in 3rd line: ', [int(b[2, i]) for i in range(0, 3, 2)])

    c = b.reshape(8, 2)
    print('5. 8x2 matrix: ', c)

    print('6. matrix multipying by scalar: ')
    scalar = int(input('enter scalar: '))
    c = c.dot(scalar)
    print(c)

    print('7: minimum in every line: ', [min(c[i]) for i in range(c.shape[0])])

    print('8. maximum in last column: ', max(c[:, c.shape[1] - 1]))

    v = numpy.array(b.reshape(1, b.shape[0] * b.shape[1]))[0].tolist()
    print('9. maximum in elements which have 0 before it: ', max(get_element_w_zero_before(v)))

    res = 1
    for i, j in zip(range(b.shape[0]), range(b.shape[1])):
        res *= b[i, j]
    print('10. result of multiplying elements from main giagonal: ', res)

t1_1()
