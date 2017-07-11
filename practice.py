import math
import re
import copy
import time
import random
import numpy
import pandas
import pylab
from matplotlib import mlab
from functools import reduce


class MatrixHelper:
    @staticmethod
    def fill_numpy_matrix_with_random(first_dimension: int, second_dimension: int,
                                      start: int, stop: int):
        return numpy.array(numpy.random.random_integers(start, stop, (first_dimension, second_dimension)))

    @staticmethod
    def fill_matrix_with_random(first_dimension: int, second_dimension: int,
                                      start: int, stop: int):
        return [[random.randint(-50, 50) for i in range(second_dimension)] for j in range(first_dimension)]

    @staticmethod
    def concatenate(first_matrix: list, second_matrix: list, axis = 0):
        if len(first_matrix[0]) == len(second_matrix[0]) and axis == 0:
            result = copy.deepcopy(first_matrix)
            for i in range(len(second_matrix)):
                result.append(second_matrix[i])
            return result
        elif len(first_matrix) == len(second_matrix):
            result = copy.deepcopy(first_matrix)
            for i in range(len(second_matrix)):
                result[i] += second_matrix[i]
            return result
        else:
            raise ValueError('matrixes have different dimensions')


class Practice:

    title = 'Python Summer practice'

    class Part1:

        def page21_part2_task1(self):
            print(math.sqrt(float(input('enter area: '))) * 4)

        def page21_part3_task1(self):
            print(max(map(float, input('enter values: ').split())))

        def page44_part1_task1(self):
            def check(x, y):
                if y < 0 or x ** 2 + y ** 2 > 9:
                    print('no')
                elif y == 0 or x ** 2 + y ** 2 == 9:
                    print('on the border')
                else:
                    print('yes')

            check(float(input('enter x: ')), float(input('enter y: ')))

        def page44_part3_task1(self):
            print(*range(1, 22, 2), sep = ' ')

        def page44_part4_task1(self):
            for i in range(4):
                print('5 ' * 6)

        def page44_part5_task1(self):
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

        def page44_part6_task1(self):
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

        def page59_part1_task1(self):
            n = int(input('enter n: '))
            print(reduce(lambda s, x: s + x ** 2, range(n + 1), 0))

        def page59_part2_task1(self):
            k = int(input('enter k: '))
            x = float(input('enter x: '))

            print(reduce(lambda s, n: s + pow(x, n) / n, range(1, k + 1), 0))

        def page59_part3_task1(self):
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

        def page88_part1_task1(self):
            a = list(int(e) for e in input('enter the array: ').split())

            print(*map(lambda x: -abs(x), a))

        def page88_part2_task1(self):
            a = list(float(e) for e in input('enter the array: ').split())
            m = max(a)

            print(len(list(filter(lambda x: x == m, a))))

        def page88_part5_task1(self):
            a = list(int(e) for e in input('enter the array: ').split())

            print(*filter(lambda x: x % 2 != 0, a))

        def page30_task1(self):
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

        def page37_part1_task1(self):
            def is_prime(x):
                if x % 2 == 0:
                    return x == 2
                d = 3
                while d * d <= x and x % d != 0:
                    d += 2
                return d * d > x

            a, b = map(int, input('enter a and b: ').split())
            print(*filter(is_prime, range(a, b + 1)))

        def page37_part2_task1(self):
            def b(n):
                if n == 1:
                    return -10
                elif n == 2:
                    return 2
                else:
                    return abs(b(n - 2)) - 6 * b(n - 1)

            n = int(input('enter n: '))
            print(b(n))

        def page24_part2_task1(self):
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

        def page4_part3_task1(self):
            regex = r'\b[A-Z]?[a-z]+\b|\b[A-Z]\b|\b[А-ЯЁ]\b|\b[А-ЯЁ]?[а-яё]+\b|\b[A-Z]+\b|\b[А-ЯЁ]+\b'
            text = re.findall(regex, input('enter text: '))
            word_to_find = input('enter word to find: ')

            print(len(list(filter(lambda x: x == word_to_find, text))))

    class Part2:

        def part2_1_1_task1(self):
            l = reduce(lambda a, x: a + [random.randrange(-50, 51)], range(20), [])

            print(*l)
            print(*filter(lambda x: x > 0 and x % 7 == 0, l))

        def part2_1_2_task1(self):
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

        def part2_1_3_task1(self):
            l = reduce(lambda a, x: a + [random.randrange(-50, 51)], range(20), [])
            min_element = min(l)

            print(*filter(lambda x: x != min_element, l))

        def part2_1_4_task1(self):
            t = tuple(reduce(lambda a, x: a + [random.randrange(-50, 51) + random.random()], range(10), []))

            print(*sorted(list(t)))

        def part2_2_5_task1(self):
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

        def part2_2_6_task1(self):
            a = set(reduce(lambda res, x: res + [random.randrange(-10, 10)], range(10), []))
            b = set(reduce(lambda res, x: res + [random.randrange(0, 11)], range(10), []))

            print(*a)
            print(*b)
            print(*(a & b))
            print(reduce(lambda res, x: res + x, a & b, 0))

        def part2_3_task1(self):
            l = list(input('enter list elements: ').split())

            d = dict(zip(l, range(len(l))))
            print(*d.items())

        def part2_3_task2(self):
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

        def part2_3_task3(self):
            l = reduce(lambda a, x: a + [random.randrange(-50, 51)], range(20), [])

            first_max = max(l)
            second_max = max(list(filter(lambda x: x != first_max, l)))
            third_max = max(list(filter(lambda x: x != first_max and x != second_max, l)))

            print(*l)
            print('first max - ', first_max)
            print('second max - ' + str(second_max))
            print('third max - ' + str(third_max))

        def part2_3_task4(self):
            regex = r'\b[A-Z]?[a-z]+\b|\b[A-Z]\b|\b[А-ЯЁ]\b|\b[А-ЯЁ]?[а-яё]+\b|\b[A-Z]+\b|\b[А-ЯЁ]+\b'
            text = re.findall(regex, input('enter text: '))
            d = {}.fromkeys(text, 0)

            for element in text:
                d[element] += 1

            print(*d.items())

    class Part3:

        class Block1:
            def task1(self):
                print(MatrixHelper.fill_numpy_matrix_with_random(4, 4, -50, 50))

            def task2(self, matrix):
                print('2. element with indexes [2, 3]: ', matrix[2, 3])

            def task3(self, matrix):
                print('3. first matrix line: ', matrix[0])

            def task4(self, matrix):
                print('4. every second element in 3rd line: ', matrix[2, ::2])

            def task5(self, matrix, new_dimensions: tuple):
                new_matrix = matrix.reshape(new_dimensions[0], new_dimensions[1])
                print('5. matrix with new dimensions: ', new_matrix)
                return new_matrix

            def task6(self, matrix):
                print('6. matrix multipying by scalar: ')
                scalar = int(input('enter scalar: '))
                matrix = matrix.dot(scalar)
                print(matrix)
                return matrix

            def task7(self, matrix):
                print('7: minimum in every line: ', [min(matrix[i]) for i in range(matrix.shape[0])])

            def task8(self, matrix):
                print('8. maximum in last column: ', max(matrix[:, matrix.shape[1] - 1]))

            def task9(self, matrix):
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

            def task10(self, matrix):
                result = 1
                for i in range(matrix.shape[0]):
                    result *= matrix[i, i]

                print('10. result of multiplying elements from main giagonal: ', result)

        class Block2:

            def task1(self):
                n, m = map(int, input('enter n and m: ').split())
                a = MatrixHelper.fill_numpy_matrix_with_random(n, m, -50, 100)
                print(a)

                maximum, index = a[0, 0], 0
                for i in range(a.shape[1]):
                    t = sum(a[:, i])
                    if maximum < t:
                        maximum = t
                        index = i
                print(max(a[:, index]))

            def task6(self):
                n, m = map(int, input('enter n and m: ').split())
                a = MatrixHelper.fill_numpy_matrix_with_random(n, m, -50, 100)
                print(a, end='\n\n')

                sum_of_columns, sum_of_all = sum(a), sum(sum(a))
                b = [numpy.array([e / (sum_of_all / 100) for e in sum_of_columns])]
                a = numpy.concatenate((a, b), 0)

                print(a)

            def task11(self):
                n, m, l = map(int, input('enter n, m and l: ').split())
                a = MatrixHelper.fill_numpy_matrix_with_random(n, m, -50, 100)
                print(a)

                for i in range(n):
                    a[i] += a[l]

                print(a)

            def task16(self):
                n, m = map(int, input('enter n and m: ').split())
                a = MatrixHelper.fill_numpy_matrix_with_random(n, m, -50, 100)
                print(a, end='\n\n')

                l = int(input('enter l: '))

                print(numpy.delete(a, l, axis=0))

            def task21(self):
                n, m = map(int, input('enter n and m: ').split())
                a = MatrixHelper.fill_numpy_matrix_with_random(n, m, -50, 100)
                print(a)

                min_dim = min((n, m))
                for i in range(min_dim - 1):
                    half_sum = (a[i + 1, i] + a[i, i + 1]) / 2
                    a[i + 1, i] = a[i, i + 1] = half_sum

                print(a)

            def task26(self):
                n, m = map(int, input('enter n and m: ').split())
                a = MatrixHelper.fill_numpy_matrix_with_random(n, m, -50, 100)
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

        def block1(self):
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

        def block2(self):
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
                'x': numpy.arange(a, b + h, h),
                'f(x)': [round(f(e), 3) for e in numpy.arange(a, b + h, h)]
            })
            print(table)

            table.loc[len(table)] = [round((f(b + 1)), 3), b + 1]
            print(table)

            table.drop(0, axis = 0, inplace = True)
            print(table)

            table = table[table.x >= (b // 2)]
            print(table)

            table.to_csv('my_own_table.csv')
            print('table loaded from created file:\n', pandas.read_csv('my_own_table.csv', index_col = 0))

    class Part5:

        def task1(self):
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

        def get_operation_statistics(self):
            def get_my_concatenate_time(first_matrix: list, second_matrix: list, axis = 0):
                start = time.time()
                MatrixHelper.concatenate(first_matrix, second_matrix, axis = axis)
                stop = time.time()
                return stop - start

            def get_numpy_concatenate_time(first_matrix: list, second_matrix: list, axis = 0):
                start = time.time()
                numpy.concatenate((first_matrix, second_matrix), axis = axis)
                stop = time.time()
                return stop - start

            def get_operation_time(first_dimension: int, second_dimension: int):
                first_matrix = list(MatrixHelper.fill_matrix_with_random(first_dimension, second_dimension, -50, 50))
                second_matrix = list(MatrixHelper.fill_matrix_with_random(first_dimension, second_dimension, -50, 0))

                return ([
                    round(get_my_concatenate_time(first_matrix, second_matrix, axis = 0), 6),
                    round(get_numpy_concatenate_time(first_matrix, second_matrix, axis = 0), 6),
                    round(get_my_concatenate_time(first_matrix, second_matrix, axis = 1), 6),
                    round(get_numpy_concatenate_time(first_matrix, second_matrix, axis = 1), 6)]
                )

            first = []
            for i in range(50):
                first.append(get_operation_time(50, 50))
            print(*first)

            first_table = pandas.DataFrame({
                'my as rows' : [e[0] for e in first],
                'numpy as rows' : [e[1] for e in first],
                'my as columns' : [e[2] for e in first],
                'numpy as columns' : [e[3] for e in first],
                'dimensions' : '50x50'
            })

            second = []
            for i in range(50):
                second.append(get_operation_time(200, 200))
            print(*second)

            second_table = pandas.DataFrame({
                'my as rows' : [e[0] for e in second],
                'numpy as rows' : [e[1] for e in second],
                'my as columns' : [e[2] for e in second],
                'numpy as columns' : [e[3] for e in second],
                'dimensions' : '200x200'
            })

            third = []
            for i in range(50):
                third.append(get_operation_time(500, 500))
            print(*third)

            third_table = pandas.DataFrame({
                'my as rows' : [e[0] for e in third],
                'numpy as rows' : [e[1] for e in third],
                'my as columns' : [e[2] for e in third],
                'numpy as columns' : [e[3] for e in third],
                'dimensions' : '500x500'
            })

            result = pandas.concat([first_table, second_table, third_table], ignore_index = True)
            pandas.DataFrame(result).to_csv('statistics.csv')

        def show_first_plot(self):
            table = pandas.read_csv('statistics.csv', index_col = 0)
            x_values = mlab.frange(1, 50, 1)
            pylab.title('50x50 matrix')
            pylab.xlabel('№')
            pylab.ylabel('Execution time')
            pylab.plot(x_values, table.loc[0:49]['my as rows'], '#ff0000')
            pylab.plot(x_values, table.loc[0:49]['my as columns'], '#00ff00')
            pylab.plot(x_values, table.loc[0:49]['numpy as rows'], '#ff00ff')
            pylab.plot(x_values, table.loc[0:49]['numpy as columns'], '#0000ff')
            pylab.show()

        def show_second_plot(self):
            table = pandas.read_csv('statistics.csv', index_col=0)
            x_values = mlab.frange(51, 100, 1)
            pylab.title('50x50 matrix')
            pylab.xlabel('№')
            pylab.ylabel('Execution time')
            pylab.plot(x_values, table.loc[50:99]['my as rows'], '#ff0000')
            pylab.plot(x_values, table.loc[50:99]['my as columns'], '#00ff00')
            pylab.plot(x_values, table.loc[50:99]['numpy as rows'], '#ff00ff')
            pylab.plot(x_values, table.loc[50:99]['numpy as columns'], '#0000ff')
            pylab.show()

        def show_third_plot(self):
            table = pandas.read_csv('statistics.csv', index_col=0)
            x_values = mlab.frange(101, 150, 1)
            pylab.title('500x500 matrix')
            pylab.xlabel('№')
            pylab.ylabel('Execution time')
            pylab.plot(x_values, table.loc[100:149]['my as rows'], '#ff0000')
            pylab.plot(x_values, table.loc[100:149]['my as columns'], '#00ff00')
            pylab.plot(x_values, table.loc[100:149]['numpy as rows'], '#ff00ff')
            pylab.plot(x_values, table.loc[100:149]['numpy as columns'], '#0000ff')
            pylab.show()

Practice.Part5.show_first_plot(Practice.Part5)
Practice.Part5.show_second_plot(Practice.Part5)
Practice.Part5.show_third_plot(Practice.Part5)
