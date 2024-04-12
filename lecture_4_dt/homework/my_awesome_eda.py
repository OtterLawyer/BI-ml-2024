import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df):
    import pip
    import sys
    try:
        __import__('cowsay')
    except ImportError:
        pip.main(['install', 'cowsay'])

    import cowsay

    cowsay.cow('Namaste')

    print(f'There are {df.shape[0]} rows and {df.shape[1]} columns')
    print('-------------')

    cols = df.columns
    DATA_TYPES_MAP = {'int64': 'int', 'object': 'str', 'float64': 'float', 'bool': 'bool',
                      'datetime64': 'datetime', 'timedelta': 'timedelta', 'category': 'categorical'}

    def get_type(col):
        if len(col) < 7 and DATA_TYPES_MAP[str(df[col].dtypes)] == 'int':
            return 'categorical'
        return DATA_TYPES_MAP[str(df[col].dtypes)]

    type_map = dict()

    for i in cols:
        type_map[i] = get_type(i)
        print(f'{i} has {df[i].dtypes} dtype which can be categorised as {type_map[i]}')

        if type_map[i] == 'categorical':
            print(df[i].value_counts())
            print((df[i].value_counts(normalize=True)))

        if type_map[i] == 'int' or type_map[i] == 'float':
            print(f'Minimal is {df[i].min()}')
            print(f'Maximum is {df[i].max()}')
            mean = df[i].mean()
            print(f'Mean is {mean}')
            q1 = df[i].quantile(0.25)
            print(f'q0.25 equals {q1}')
            print(f'Median is {df[i].median()}')
            q3 = df[i].quantile(0.75)
            print(f'q0.75 equals {q3}')
            iqr = q3 - q1
            print(f'There are {len(df[i][df[i] < mean - iqr * 1.5]) + len(df[i][df[i] > mean + iqr * 1.5])} outliers')
        print('-------------')

    print(
        f'There are {sum(df.isna().sum())} of NaN values in {df.shape[0] - df.dropna().shape[0]} rows in columns {df.columns[df.isna().any()].tolist()} in dataframe ')

    print(f'There are duplicates in {len(df.groupby(df.columns.tolist(), as_index=False).size())} rows')

    # Plotting the percentage of missing values in each column
    missing_values = df.isnull().mean() * 100
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_values.index, y=missing_values.values)
    plt.title('Percentage of missing values in each column')
    plt.ylabel('Missing values (%)')
    plt.xticks(rotation=90)
    plt.show()

    # Plotting the correlation heatmap for all variables
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(), annot=True, square=True, cmap='coolwarm')
    plt.title('Correlation heatmap')
    plt.show()

    # Plotting the histogram and boxplot for each numeric variable
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        sns.histplot(df[col], kde=False)
        plt.title(f'Distribution of {col}')
        plt.subplot(212)
        sns.boxplot(df[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()
