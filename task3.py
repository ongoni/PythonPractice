#p.44, I, 1)
x, y = map(float, input('enter x and y: ').split())
if y < 0 or x**2 + y**2 > 9:
    print('no')
elif y == 0 or x**2 + y**2 == 9:
    print('on edge')
else:
    print('yes')
