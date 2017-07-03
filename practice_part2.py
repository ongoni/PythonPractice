import random
from functools import reduce

def t1_1():
    l = reduce(lambda a, x: a + [random.randrange(-50, 51)], range(20), [])

    print(*l)
    print(*filter(lambda x: x > 0 and x % 7 == 0, l))

def t2_1():
    l = reduce(lambda a, x: a + [random.randrange(-10, 130)], range(20), [])

    to_insert = int(input('enter x: '))

    result = reduce(
        lambda res, x:
            res + [x, to_insert] if 0 < abs(x // 10) < 10
            else res + [x],
        l,
        []
    )

    print(*result)

def t3_1():
    l = reduce(lambda a, x: a + [random.randrange(-50, 51)], range(20), [])
    min_element = min(l)

    print(*filter(lambda x: x != min_element, l))

def t4_1():
    t = tuple(reduce(lambda a, x: a + [random.randrange(-50, 51) + random.random()], range(10), []))

    print(*sorted(list(t)))

def t5_1():
    t = tuple(reduce(lambda res, x: res + [random.randrange(-50, 51)], range(10), []))
    a, b = map(int, input('enter a and b: ').split())

    result = reduce(
        lambda res, x:
            res + [t.index(x)] if a <= x <= b
            else res,
        list(t),
        []
    )

    print(*t)
    print(*result)

def t6_1():
    a = set(reduce(lambda res, x: res + [random.randrange(-10, 10)], range(10), []))
    b = set(reduce(lambda res, x: res + [random.randrange(0, 11)], range(10), []))

    print(*a)
    print(*b)
    print(*(a & b))
    print(reduce(lambda res, x: res + x, a & b, 0))

# t1_1()
# t2_1()
# t3_1()
# t4_1()
# t5_1()
t6_1()
