# p.59, III, 1)
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
