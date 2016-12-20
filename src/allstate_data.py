import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler
import colored_traceback.always

class AllStateData(object):
    def __init__(self, datadir = 'input', include = [], exclude = [], **featureparams):
        self.data = self.load_data(datadir)
        self.trainids= self.data['train'].index
        self.testids = self.data['test'].index
        self.preprocessData(**featureparams)
        self.include = include
        self.exclude = exclude
        
    def load_data(self, datadir):
        nrows = None 
        train = pd.read_csv(os.path.join(datadir, 'train.csv'),
                            index_col='id', nrows=nrows) 
        test  = pd.read_csv(os.path.join(datadir, 'test.csv'),
                            index_col='id', nrows=nrows) 
        return dict(train=train, test=test)
    
    def preprocessData(self, shift = 200.0):
        train = self.data['train']
        test  = self.data['test']
        test['loss'] = np.nan
        df = pd.concat([train, test],ignore_index=True)
        dfcats = [c for c in df.columns if "cat" in c]
        for c in dfcats:
            if train[c].nunique() != test[c].nunique():
                set_train = set(train[c].unique())
                set_test = set(test[c].unique())
                remove_train = set_train - set_test
                remove_test = set_test - set_train
                remove = remove_train.union(remove_test)
                def filter_cat(x):
                    if x in remove:
                        return np.nan
                    return x    
                df[c] = df[c].apply(lambda x: filter_cat(x), 1)
            df[c] = pd.factorize(df[c].values,sort=True)[0] 
        for c in df.columns:
            if "cont" in c:
                ss = StandardScaler()
                df.loc[df['loss'].notnull(),c] = ss.fit_transform(df[df['loss'].notnull()][c].values.reshape(-1,1))
                df.loc[df['loss'].isnull(), c] = ss.transform(df[df['loss'].isnull()][c].values.reshape(-1,1))
        features = [f for f in df.columns if f not in ['loss']]
        train = df[df['loss'].notnull()]
        test = df[df['loss'].isnull()]
        self.Xtr = train.drop(['loss'], 1)
        self.ytr = pd.DataFrame(np.log(train['loss'].values+shift))
        self.Xte = test.drop(['loss'], 1)
        self.features = features
        self.data = None
        self.shift = shift
    
    def inverseScaleTransform(self, y):
        return (np.exp(y)-self.shift)
    
    def mae(self, ypred, ytrue):
        return mean_absolute_error(self.inverseScaleTransform(ytrue), 
                                   self.inverseScaleTransform(ypred))
    
    def get_train_test_features(self, itrain = None, itest = None):
        if itrain is None and itest is None:
            # train-test splitting
            Xtr = self.Xtr
            ytr = self.ytr
            Xte = self.Xte
            yte = None
        elif itrain is not None and itest is not None: 
            # train-validation splitting
            itrain = self.Xtr.index[itrain]
            itest  = self.Xtr.index[itest]
            Xtr = self.Xtr.loc[itrain, self.features]
            ytr = self.ytr.loc[itrain]
            Xte = self.Xtr.loc[itest , self.features]
            yte = self.ytr.loc[itest]
        else:
            raise RuntimeError('Wrong usage of get_train_test_features')
        # Include or exclude features
        if len(self.include) > 0:
            usefeatures = list(self.include)
        else:
            usefeatures = list(Xtr.columns)
        usefeatures = [f for f in usefeatures if f not in self.exclude]
        return Xtr[usefeatures], ytr, Xte[usefeatures], yte
