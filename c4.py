import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

titanic = pd.read_csv('./titanic.csv')
#print titanic

print titanic.head()[['pclass', 'survived', 'age', 'embarked', 'boat', 'sex']]

from sklearn import feature_extraction
def one_hot_dataframe(data, cols, replace=False):
    vec = feature_extraction.DictVectorizer