import pandas
import numpy

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
        'x' : numpy.arange(a, b, h),
        'f(x)' : [round(f(e), 3) for e in numpy.arange(a, b, h)]
    })

    print(table)

    table.loc[len(table)] = [round((f(b + 1)), 3), b + 1]

    print(table)

    table.drop(0, axis = 0, inplace = True)

    print(table)

    table = table[table.x >= (b // 2)]

    print(table)

    table.to_csv('my_own_table.csv')

    print(pandas.read_csv('my_own_table.csv'))

block2()
