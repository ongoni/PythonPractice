#p.44, VI, 1)
def func(x):
    if x >= 0.9:
        return 1.0 / (0.1 + x) ** 2
    elif x >= 0:
        return 0.2 * x + 0.1
    else:
        return x ** 2 + 0.2

a, b, h = map(float, input('enter a, b and h: ').split())

while a <= b:
    print(func(a))
    a += h
