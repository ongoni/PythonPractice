import pylab
from matplotlib import mlab

def f(x):
    if x >= 0.9:
        return 1.0 / (0.1 + x) ** 2
    elif x >= 0:
        return 0.2 * x + 0.1
    else:
        return x ** 2 + 0.2

def t1():
    xmin, xmax, step = map(float, input('enter xmin, xmax and step: ').split())
    x_values = mlab.frange(xmin, xmax, step)
    y_values = [f(x) for x in x_values]
    pylab.plot(x_values, y_values)
    pylab.show()

t1()
