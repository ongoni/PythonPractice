import math
from functools import reduce

def p21_II_1():
    print(math.sqrt(float(input('enter area: '))) * 4)

def p21_III_1():
    print(max(map(float, input('enter values: ').split())))

def p44_I_1():
    def check(x, y):
        if y < 0 or x ** 2 + y ** 2 > 9:
            print('no')
        elif y == 0 or x ** 2 + y ** 2 == 9:
            print('on the border')
        else:
            print('yes')

    check(float(input('enter x: ')), float(input('enter y: ')))

def p44_III_1():
    print(*range(1, 22, 2), sep = ' ')

def p44_IV_1():
    for i in range(4):
        print('5 ' * 6)

def p44_V_1():
    def f(x):
        if x == -1:
            print('function is not defined in x = -1')
        else:
            print(str(1.0 / (1 + x) ** 2))

    a, b, h = map(float, input('enter a, b and h: ').split())

    while a <= b:
        f(a)
        a += h

def p44_VI_1():
    def f(x):
        if x >= 0.9:
            return 1.0 / (0.1 + x) ** 2
        elif x >= 0:
            return 0.2 * x + 0.1
        else:
            return x ** 2 + 0.2

    a, b, h = map(float, input('enter a, b and h: ').split())

    while a <= b:
        print(f(a))
        a += h

def p59_I_1():
    n = int(input('enter n: '))

    print(reduce(lambda s, x: s + x ** 2, range(n + 1), 0))

def p59_II_1():
    k = int(input('enter k: '))
    x = float(input('enter x: '))

    print(reduce(lambda s, n: s + pow(x, n) / n, range(1, k + 1), 0))

def p59_III_1():
    def sum(e):
        s = 0
        i = 1
        while True:
            t = 1 / (i ** 2)
            if t < e:
                break
            s += t
            i += 1
        return s

    e = float(input('enter e: '))

    print(sum(e))

def p88_I_1():
    a = list(int(e) for e in input('enter the array: ').split())

    print(*map(lambda x: -abs(x), a))

def p88_II_1():
    a = list(int(e) for e in input('enter the array: ').split())
    m = max(a)

    print(len(list(filter(lambda x: x == m, a))))

def p88_V_1():
    a = list(int(e) for e in input('enter the array: ').split())

    print(*filter(lambda x: x % 2 != 0, a))

def p30_1():
    def max(a, b):
        if a > b or a == b:
            return a
        else:
            return b

    def min(a, b):
        if a > b or a == b:
            return b
        else:
            return a

    x, y = map(int, input('enter a and b: ').split())

    print(str(min(3 * x, 2 * y) + min(x - y, x + y)))

p30_1()
