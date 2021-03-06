import pandas as pd
import os,sys,inspect
import sys
sys.path.append("..")
from autopredict.classification import autoClassify
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from datasets import loadiris
#from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import grid
import time
from scipy.stats import pearsonr




if __name__ == '__main__':
    pd.set_option('display.max_columns',50000)
    pd.set_option('display.width', 500000)
    df_train = pd.read_csv('./tests/train.csv',encoding="ISO-8859-1")
    df_test = pd.read_csv('./tests/test.csv',encoding="ISO-8859-1")

    df_train['discount'] = df_train['discount_amount'].apply(lambda x: 1
    if x > 0 else 0)

    df_train['total'] = df_train['fine_amount'] + df_train['admin_fee'] \
                        + df_train['state_fee'] + df_train['late_fee']

    df_train['total_bin']= pd.cut(x=df_train['total'],
                                  bins =[0,100,200,300,400,500,600
                                         ,700,800,900,1000,40000],
                                  labels=[0,1,2,3,4,5,6,7,8,9,10])

    df_train = df_train[df_train['compliance'].notnull()]
    #print(df_train.isna().any().sum())
    cols = ['agency_name', 'discount', 'state', 'disposition', 'total_bin', 'compliance']
    df_train = df_train[cols]
    #print(df_train.isna().any())
    df_train['compliance'] = df_train['compliance'].astype(int)



    #print(df_train.columns)
    X = df_train.drop('compliance',axis=1)
    y = df_train['compliance']


    X['state'].fillna('MI',inplace=True)

    #print(grid.getClassificationGridDict())
    start = time.time()
    ## for using grid parameter search
    #tmp =autoClassify(encoder='label',useGridtuning=True)
    tmp = autoClassify(encoder='label', useGridtuning=False)
    tmp.fit(X,y)
    print(tmp._predict_df)
    print(time.time()-start)

    print(X.dtypes)
    print(pearsonr(X.discount,y))
