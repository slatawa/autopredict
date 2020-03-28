import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.linear_model import LogisticRegression

def reduce_memory(df):
    """
    This method can be used to reduce memory usage of a dataframe by compacting int/float columns
    :param df: pass the input dataframe that you want to compress
    :return:  modified data frame with compressed features
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    logging.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    logging.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logging.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def rankDict(ranks,names,order=-1):
    if order == -1:
        _ = [(rec[0],idx+1) for idx,rec in enumerate(sorted(zip(names, map(lambda x: round(x, 6), ranks)),
                                                         key = lambda x : x[1],reverse=True))]
        return dict(_)
    else:
       return dict(zip(names,map(lambda x: round(x, 6), ranks)))

def rankFeatures(X,y,objective='Classify'):
    if (X.isna().any().sum() or y.isna().any().sum()):
        logging.error('Auto predict - rankFeatures does not support Nan''s'' for ranking Features' )
        exit()

    if X.shape[0] != y.shape[0]:
        logging.warning('X and y values passed should have same number of rows')
        exit()

    # Run this through label encoder to conver non-numeric features to numeric values
    for rec in X.columns:
        if (X[rec].dtype.name == 'object' or  X[rec].dtype.name == 'category'):
            X[rec] = LabelEncoder().fit_transform(X[rec])

    # for rec in y.columns:
    #     if (y[rec].dtype.name == 'object' or  y[rec].dtype.name == 'category'):
    #         y[rec] = LabelEncoder().fit_transform(y[rec])

    if objective=='Classify':
        ### apply RFE using LR
        lr = LogisticRegression()
        rfe = RFE(lr)
        rfe.fit(X,y)
        rankD ={}
        rankD['LR']= rankDict(rfe.ranking_,X.columns,order=0)
        ## Apply tandom forect classifier
        rf = RandomForestClassifier()
        rf.fit(X,y)
        rankD['RFE'] = rankDict(rf.feature_importances_, X.columns)
        # ### Apply Lasso
        # la = Lasso()
        # la.fit(X,y)
        # rankD['Lasso'] = rankDict(map(lambda x : x if x > 0 else x*-1,la.coef_), X.columns)
    elif objective == 'Regression':
        ### apply RFE using LR
        lr = LinearRegression()
        rfe = RFE(lr)
        rfe.fit(X, y)
        rankD = {}
        rankD['LR'] = rankDict(rfe.ranking_, X.columns, order=0)
        ## Apply tandom forect classifier
        rf = RandomForestClassifier()
        rf.fit(X, y)
        rankD['RFE'] = rankDict(rf.feature_importances_, X.columns)
        ### Apply Lasso
        la = Lasso()
        la.fit(X, y)
        rankD['Lasso'] = rankDict(map(lambda x: x if x > 0 else x * -1, la.coef_), X.columns)
    else:
        logging.error('Incorrect value choosen for Objective parameter - please refer the'
                      'documentation ')
        exit('Exited')

    ### rankD dict should be ready at this point
    ### convert this into a readable information - sorting column names importance with 1 being the most important
    finalRank = {}
    for name in X.columns:
        finalRank[name] = round(np.mean([rankD[method][name] for method in rankD.keys()]), 2)
    return pd.DataFrame(sorted(finalRank.items(), key=lambda x: x[1]), columns=['Column-name', 'Importance-Rank'])



