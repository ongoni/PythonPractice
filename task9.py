# p.59, II, 1)
def sum(k, x):
    s = 0
    i = 1
    t = x
    while i <= k:
        s += t / i
        t = t * x
        i += 1
    return s

k = int(input('enter k: '))
x = float(input('enter x: '))

print(sum(k, x))
