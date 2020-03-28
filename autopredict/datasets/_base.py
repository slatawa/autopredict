import pandas as pd
import os

def loadiris():
    print(os.getcwd())
    return pd.read_csv('./tests/iris.csv')