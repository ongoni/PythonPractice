import random
import re
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

def t1():
    l = list(input('enter list elements: ').split())

    d = dict(zip(l, range(len(l))))
    print(*d.items())

def t2():
    line = input('enter line: ').split()
    d = { 'lol' : 'kek', 'kek' : 'lol'}

    result = reduce(
        lambda res, x:
            res + [d[x]] if x in d
            else res + [x],
        line,
        []
    )

    print(result)

def t3():
    l = reduce(lambda a, x: a + [random.randrange(-50, 51)], range(20), [])

    first_max = max(l)
    second_max = max(list(filter(lambda x: x != first_max, l)))
    third_max = max(list(filter(lambda x: x != first_max and x != second_max, l)))

    print(*l)
    print('first max - ', first_max)
    print('second max - ' + str(second_max))
    print('third max - ' + str(third_max))

def t4():
    text = re.findall(r'\b[A-Z]?[a-z]+\b|\b[A-Z]\b|\b[А-ЯЁ]\b|\b[А-ЯЁ]?[а-яё]+\b|\b[A-Z]+\b|\b[А-ЯЁ]+\b',
                      input('enter text: '))
    d = {}.fromkeys(text, 0)

    for element in text:
        d[element] += 1

    print(*d.items())

