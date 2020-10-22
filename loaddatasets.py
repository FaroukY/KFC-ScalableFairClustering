from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd


def blobs_dataset():
    k=10
    C = 20
    N = 5000
    X,_ = make_blobs(N, centers=k, random_state=3, center_box=(-100,100))
    colors = np.random.randint(0,C, len(X))
    groups = {i : [] for i in range(C)}
    for i,c in enumerate(colors):
        groups[c].append(i)
    
    return X, k, groups

def get_reuters():
    df = pd.read_csv('data/c50.csv')
    C = np.array(df.iloc[:,1:])
    colors = df['color']
    groups = {}
    for k in range(df['color'].max()+1):
        groups[k] = df.loc[df['color'] == k].index.tolist()

    return C, groups, colors, 'c50'

def get_victorian():
    df = pd.read_csv('data/victorian.csv')
    C = np.array(df.iloc[:,1:])
    colors = df['color']
    groups = {}
    for k in range(df['color'].max()+1):
        groups[k] = df.loc[df['color'] == k].index.tolist()

    return C, groups, colors, 'victorian'

def get_4area():
    df = pd.read_csv('data/4area.csv')
    C = np.array(df.iloc[:,1:])
    colors = df['color']
    groups = {}
    
    for k in range(df['color'].max()+1):
        groups[k] = df.loc[df['color'] == k].index.tolist()

    return C, groups, colors, '4area'

def get_bank():
    df = pd.read_csv('data/bank_categorized.csv')
    C = np.array(df[['age', 'balance', 'duration', 'marital', 'default']])
    
    groups = {}
    for k in range(df['marital'].max()+1):
        groups[k] = df.loc[df['marital'] == k].index.tolist()
    for k in range(df['marital'].max()+1, df['marital'].max()+1+df['default'].max()+1):
       groups[k] = df.loc[df['default'] == (k - df['marital'].max()+1)].index.tolist()

    return C, groups, None, 'bank'

def get_census():
    df = pd.read_csv('data/adult_categorized.csv')

    C = np.array(df[['age', 'final-weight', 'education-num', 'capital-gain', 'hours-per-week']])
    groups = {}
    for k in range(df['race'].max()+1):
        groups[k] = df.loc[df['race'] == k].index.tolist()
    for k in range(df['race'].max()+1, df['race'].max()+1+df['sex'].max()+1):
        groups[k] = df.loc[df['sex'] == (k - df['race'].max()+1)].index.tolist()

    return C, groups, None, 'adult'

def get_creditcard():
    df = pd.read_csv('data/creditcard.csv')
    C = np.array(df.drop(['ID', 'SEX', 'EDUCATION', 'MARRIAGE', 'default payment next month'], axis=1))
    
    groups = {}
    for k in range(df['MARRIAGE'].max()+1):
        groups[k] = df.loc[df['MARRIAGE'] == k].index.tolist()
    for k in range(1, 5):
        if k == 1:
            groups[k+df['MARRIAGE'].max()] = df.loc[df['EDUCATION'] <= 1].index.tolist()
        elif k == 4:
            groups[k+df['MARRIAGE'].max()] = df.loc[df['EDUCATION'] >= 4].index.tolist()
        else:
            groups[k+df['MARRIAGE'].max()] = df.loc[df['EDUCATION'] == k].index.tolist()

    return C, groups, None, 'creditcard'