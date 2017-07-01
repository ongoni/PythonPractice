#V, 1)
def func(x):
    if x == -1:
        print('function is not defined in x = -1')
        return 0
    else:
        return 1.0 / ((1 + x) ** 2)

a, b, h = map(float, input('enter a, b and h: ').split())

while a <= b:
    print(func(a))
    a += h
