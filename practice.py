import math
import re
import random
import numpy
import pandas
import pylab
from matplotlib import mlab
from functools import reduce


class MatrixFiller:
    @staticmethod
    def fill_matrix_with_random(first_dimension: int, second_dimension: int,
                                start: int, stop: int):
        return numpy.array(numpy.random.random_integers(start, stop, (first_dimension, second_dimension)))


class Practice:

    title = 'Python Summer practice'

    class Part1:
        @staticmethod
        def page21_part2_task1():
            print(math.sqrt(float(input('enter area: '))) * 4)

        @staticmethod
        def page21_part3_task1():
            print(max(map(float, input('enter values: ').split())))

        @staticmethod
        def page44_part1_task1():
            def check(x, y):
                if y < 0 or x ** 2 + y ** 2 > 9:
                    print('no')
                elif y == 0 or x ** 2 + y ** 2 == 9:
                    print('on the border')
                else:
                    print('yes')

            check(float(input('enter x: ')), float(input('enter y: ')))

        @staticmethod
        def page44_part3_task1():
            print(*range(1, 22, 2), sep = ' ')

        @staticmethod
        def page44_part4_task1():
            for i in range(4):
                print('5 ' * 6)

        @staticmethod
        def page44_part5_task1():
            def f(x):
                if x == -1:
                    print('function is not defined in x = -1')
                else:
                    print(str(1.0 / (1 + x) ** 2))

            a, b = map(float, input('enter a and b: ').split())
            h = float(input('enter h: '))

            while a <= b:
                f(a)
                a += h

        @staticmethod
        def page44_part6_task1():
            def f(x):
                if x >= 0.9:
                    return 1.0 / (0.1 + x) ** 2
                elif x >= 0:
                    return 0.2 * x + 0.1
                else:
                    return x ** 2 + 0.2

            a, b = map(float, input('enter a and b: ').split())
            h = float(input('enter h: '))

            while a <= b:
                print(f(a))
                a += h

        @staticmethod
        def page59_part1_task1():
            n = int(input('enter n: '))
            print(reduce(lambda s, x: s + x ** 2, range(n + 1), 0))

        @staticmethod
        def page59_part2_task1():
            k = int(input('enter k: '))
            x = float(input('enter x: '))

            print(reduce(lambda s, n: s + pow(x, n) / n, range(1, k + 1), 0))

        @staticmethod
        def page59_part3_task1():
            def s(e):
                s, i = 0, 1
                while True:
                    t = 1 / (i ** 2)
                    if abs(t) < e:
                        break
                    s += t
                    i += 1
                return s

            def reduce_s(e):
                return reduce(
                    lambda res, x:
                    (res + 1.0 / pow(x, 2)) if 1.0 / pow(x, 2) >= e
                    else res + 0,
                    range(1, 10000),
                    0
                )

            e = float(input('enter e: '))

            print('simple function: ', s(e))
            print('reduce: ', reduce_s(e))

        @staticmethod
        def page88_part1_task1():
            a = list(int(e) for e in input('enter the array: ').split())

            print(*map(lambda x: -abs(x), a))

        @staticmethod
        def page88_part2_task1():
            a = list(float(e) for e in input('enter the array: ').split())
            m = max(a)

            print(len(list(filter(lambda x: x == m, a))))

        @staticmethod
        def page88_part5_task1():
            a = list(int(e) for e in input('enter the array: ').split())

            print(*filter(lambda x: x % 2 != 0, a))

        @staticmethod
        def page30_task1():
            def maximum(a, b):
                if a >= b:
                    return a
                return b

            def minimum(a, b):
                if a >= b:
                    return b
                return a

            x, y = map(int, input('enter x and y: ').split())

            print((minimum(3 * x, 2 * y) + minimum(x - y, x + y)))

        @staticmethod
        def page37_part1_task1():
            def is_prime(x):
                if x % 2 == 0:
                    return x == 2
                d = 3
                while d * d <= x and x % d != 0:
                    d += 2
                return d * d > x

            a, b = map(int, input('enter a and b: ').split())
            print(*filter(is_prime, range(a, b + 1)))

        @staticmethod
        def page37_part2_task1():
            def b(n):
                if n == 1:
                    return -10
                elif n == 2:
                    return 2
                else:
                    return abs(b(n - 2)) - 6 * b(n - 1)

            n = int(input('enter n: '))
            print(b(n))

        @staticmethod
        def page24_part2_task1():
            line = input('enter line: ')
            to_find = input('enter symbol to find: ')
            to_insert = input('enter symbol to insert: ')

            result = reduce(
                lambda res, x:
                res + x + to_insert if x == to_find
                else res + x,
                line,
                ''
            )

            print(result)

        @staticmethod
        def page4_part3_task1():
            regex = r'\b[A-Z]?[a-z]+\b|\b[A-Z]\b|\b[А-ЯЁ]\b|\b[А-ЯЁ]?[а-яё]+\b|\b[A-Z]+\b|\b[А-ЯЁ]+\b'
            text = re.findall(regex, input('enter text: '))
            word_to_find = input('enter word to find: ')

            print(len(list(filter(lambda x: x == word_to_find, text))))

    class Part2:
        @staticmethod
        def part2_1_1_task1():
            l = reduce(lambda a, x: a + [random.randrange(-50, 51)], range(20), [])

            print(*l)
            print(*filter(lambda x: x > 0 and x % 7 == 0, l))

        @staticmethod
        def part2_1_2_task1():
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

        @staticmethod
        def part2_1_3_task1():
            l = reduce(lambda a, x: a + [random.randrange(-50, 51)], range(20), [])
            min_element = min(l)

            print(*filter(lambda x: x != min_element, l))

        @staticmethod
        def part2_1_4_task1():
            t = tuple(reduce(lambda a, x: a + [random.randrange(-50, 51) + random.random()], range(10), []))

            print(*sorted(list(t)))

        @staticmethod
        def part2_2_5_task1():
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

        @staticmethod
        def part2_2_6_task1():
            a = set(reduce(lambda res, x: res + [random.randrange(-10, 10)], range(10), []))
            b = set(reduce(lambda res, x: res + [random.randrange(0, 11)], range(10), []))

            print(*a)
            print(*b)
            print(*(a & b))
            print(reduce(lambda res, x: res + x, a & b, 0))

        @staticmethod
        def part2_3_task1():
            l = list(input('enter list elements: ').split())

            d = dict(zip(l, range(len(l))))
            print(*d.items())

        @staticmethod
        def part2_3_task2():
            line = input('enter line: ').split()
            d = {'lol': 'kek', 'kek': 'lol'}

            result = reduce(
                lambda res, x:
                res + [d[x]] if x in d
                else res + [x],
                line,
                []
            )

            print(result)

        @staticmethod
        def part2_3_task3():
            l = reduce(lambda a, x: a + [random.randrange(-50, 51)], range(20), [])

            first_max = max(l)
            second_max = max(list(filter(lambda x: x != first_max, l)))
            third_max = max(list(filter(lambda x: x != first_max and x != second_max, l)))

            print(*l)
            print('first max - ', first_max)
            print('second max - ' + str(second_max))
            print('third max - ' + str(third_max))

        @staticmethod
        def part2_3_task4():
            regex = r'\b[A-Z]?[a-z]+\b|\b[A-Z]\b|\b[А-ЯЁ]\b|\b[А-ЯЁ]?[а-яё]+\b|\b[A-Z]+\b|\b[А-ЯЁ]+\b'
            text = re.findall(regex, input('enter text: '))
            d = {}.fromkeys(text, 0)

            for element in text:
                d[element] += 1

            print(*d.items())

    class Part3:

        class Block1:
            @staticmethod
            def task1():
                print(MatrixFiller.fill_matrix_with_random(4, 4, -50, 50))

            @staticmethod
            def task2(matrix):
                print('2. element with indexes [2, 3]: ', matrix[2, 3])

            @staticmethod
            def task3(matrix):
                print('3. first matrix line: ', matrix[0])

            @staticmethod
            def task4(matrix):
                print('4. every second element in 3rd line: ', matrix[2, ::2])

            @staticmethod
            def task5(matrix, new_dimensions: tuple):
                new_matrix = matrix.reshape(new_dimensions[0], new_dimensions[1])
                print('5. matrix with new dimensions: ', new_matrix)
                return new_matrix

            @staticmethod
            def task6(matrix):
                print('6. matrix multipying by scalar: ')
                scalar = int(input('enter scalar: '))
                matrix = matrix.dot(scalar)
                print(matrix)
                return matrix

            @staticmethod
            def task7(matrix):
                print('7: minimum in every line: ', [min(matrix[i]) for i in range(matrix.shape[0])])

            @staticmethod
            def task8(matrix):
                print('8. maximum in last column: ', max(matrix[:, matrix.shape[1] - 1]))

            @staticmethod
            def task9(matrix):
                def get_elements_w_zero_before(vec):
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

                v = numpy.array(matrix.reshape(1, matrix.shape[0] * matrix.shape[1]))[0].tolist()
                print('9. maximum in elements which have 0 before it: ', max(get_elements_w_zero_before(v)))

            @staticmethod
            def task10(matrix):
                result = 1
                for i in range(matrix.shape[0]):
                    result *= matrix[i, i]

                print('10. result of multiplying elements from main giagonal: ', result)

        class Block2:
            @staticmethod
            def task1():
                n, m = map(int, input('enter n and m: ').split())
                a = MatrixFiller.fill_matrix_with_random(n, m, -50, 100)
                print(a)

                maximum, index = a[0, 0], 0
                for i in range(a.shape[1]):
                    t = sum(a[:, i])
                    if maximum < t:
                        maximum = t
                        index = i
                print(max(a[:, index]))

            @staticmethod
            def task6():
                n, m = map(int, input('enter n and m: ').split())
                a = MatrixFiller.fill_matrix_with_random(n, m, -50, 100)
                print(a, end='\n\n')

                sum_of_columns, sum_of_all = sum(a), sum(sum(a))
                b = [numpy.array([e / (sum_of_all / 100) for e in sum_of_columns])]
                a = numpy.concatenate((a, b), 0)

                print(a)

            @staticmethod
            def task11():
                n, m, l = map(int, input('enter n, m and l: ').split())
                a = MatrixFiller.fill_matrix_with_random(n, m, -50, 100)
                print(a)

                for i in range(n):
                    a[i] += a[l]

                print(a)

            @staticmethod
            def task16():
                n, m = map(int, input('enter n and m: ').split())
                a = MatrixFiller.fill_matrix_with_random(n, m, -50, 100)
                print(a, end='\n\n')

                l = int(input('enter l: '))

                print(numpy.delete(a, l, axis=0))

            @staticmethod
            def task21():
                n, m = map(int, input('enter n and m: ').split())
                a = MatrixFiller.fill_matrix_with_random(n, m, -50, 100)
                print(a)

                min_dim = min((n, m))
                for i in range(min_dim - 1):
                    half_sum = (a[i + 1, i] + a[i, i + 1]) / 2
                    a[i + 1, i] = a[i, i + 1] = half_sum

                print(a)

            @staticmethod
            def task26():
                n, m = map(int, input('enter n and m: ').split())
                a = MatrixFiller.fill_matrix_with_random(n, m, -50, 100)
                print(a)

                l, k = map(int, input('enter l and k: ').split())

                first = a[:l, :k]
                print(first, sep='\n', end='\n')
                print('average in first part: ',
                      sum(sum(first)) /
                      (first.shape[0] * first.shape[1]))

                second = a[:l, k:]
                print(second, sep='\n', end='\n')
                print('average in second part: ',
                      sum(sum(second)) /
                      (second.shape[0] * second.shape[1]))

                third = a[l:, :k]
                print(third, sep='\n', end='\n')
                print('average in third part: ',
                      sum(sum(third)) /
                      (third.shape[0] * third.shape[1]))

                fourth = a[l:, k:]
                print(fourth, sep='\n', end='\n')
                print('average in fourth part: ',
                      sum(sum(fourth)) /
                      (fourth.shape[0] * fourth.shape[1]))

    class Part4:
        @staticmethod
        def block1():
            df = pandas.read_csv('titanic.csv')
            print(df)

            print(len(df[df.Survived == 0]), ' not survived')

            women_children = df[((df.Sex == 'female') & (df.Age >= 18))
                                | (df.Age < 18)]
            women_children_survived = women_children[women_children.Survived == 1]
            print(round(len(women_children_survived) / (len(women_children) / 100), 2),
                  '% of women and children survived')

            men = df[(df.Sex == 'male') & (df.Age >= 18)]
            men_survived = men[men.Survived == 1]
            print(round(len(men_survived) / (len(men) / 100), 2),
                  '% of men survived')

            first_class = df[df.Pclass == 1]
            print(round(len(first_class) / (len(df) / 100), 2),
                  '% of passengers were in 1st class')

            children = df[df.Age < 18]
            print(round(len(children) / (len(df) / 100), 2),
                  '% of passengers were children')

        @staticmethod
        def block2():
            def f(x):
                if x >= 0.9:
                    return 1.0 / (0.1 + x) ** 2
                elif x >= 0:
                    return 0.2 * x + 0.1
                else:
                    return x ** 2 + 0.2

            a, b = map(int, input('enter a and b: ').split())
            h = float(input('enter h: '))

            table = pandas.DataFrame({
                'x': numpy.arange(a, b, h),
                'f(x)': [round(f(e), 3) for e in numpy.arange(a, b, h)]
            })
            print(table)

            table.loc[len(table)] = [round((f(b + 1)), 3), b + 1]
            print(table)

            table.drop(0, axis=0, inplace=True)
            print(table)

            table = table[table.x >= (b // 2)]
            print(table)

            table.to_csv('my_own_table.csv')
            print('table loaded from created file:\n', pandas.read_csv('my_own_table.csv', index_col=0))

    class Part5:
        @staticmethod
        def task1():
            def f(x):
                if x >= 0.9:
                    return 1.0 / (0.1 + x) ** 2
                elif x >= 0:
                    return 0.2 * x + 0.1
                else:
                    return x ** 2 + 0.2

            xmin, xmax, step = map(float, input('enter xmin, xmax and step: ').split())
            x_values = mlab.frange(xmin, xmax, step)
            y_values = [f(x) for x in x_values]
            pylab.plot(x_values, y_values)
            pylab.show()
