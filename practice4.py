import pandas

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
    a, b, h = map(int, input('enter a, b abd h: ').split())

block2()
