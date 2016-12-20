from __future__ import division
import os
import pandas as pd
import numpy as np
import datetime
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import sys
sys.path = [''] + sys.path 
from src.allstate_data import AllStateData
from sacred import Experiment
from sacred.observers import MongoObserver
import colored_traceback.always
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
ex = Experiment('kerasRun')
@ex.config
def my_config():
    output_path = "keras"
    datadir = 'input'
    n_folds = 10
    clfparams = {}
    include = []
    exclude = []
    featureparams = {'shift':200.0}
    save_oob_predictions  = False
    save_test_predictions = True
    skip_cross_validation = False
    
@ex.main
def experiment(output_path,
               datadir,
               n_folds,
               clfparams,
               include,
               exclude,
               featureparams,
               save_oob_predictions,
               save_test_predictions,
               skip_cross_validation,
               _run):
    ## Batch generators ##################################################################################################################################

    def batch_generator(X, y, batch_size, shuffle):
        #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
        number_of_batches = np.ceil(X.shape[0]/batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(sample_index)
        while True:
            batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
            X_batch = X[batch_index,:].toarray()
            y_batch = y[batch_index]
            counter += 1
            yield X_batch, y_batch
            if (counter == number_of_batches):
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0
    
    def batch_generatorp(X, batch_size, shuffle):
        number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        while True:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
            X_batch = X[batch_index, :].toarray()
            counter += 1
            yield X_batch
            if (counter == number_of_batches):
                counter = 0
                
    def evalerror(preds, labels):
        return mean_absolute_error(
            np.exp(preds), np.exp(labels))
    
    ########################################################################################################################################################
    
    ## read data
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')
    
    ## set test loss to NaN
    test['loss'] = np.nan
    
    ## response and IDs
    y = np.log(train['loss'].values+featureparams['shift'])
    id_train = train['id'].values
    id_test = test['id'].values
    
    ## stack train test
    ntrain = train.shape[0]
    tr_te = pd.concat((train, test), axis = 0)
    
    ## Preprocessing and transforming to sparse data
    sparse_data = []
    
    f_cat = [f for f in tr_te.columns if 'cat' in f]
    for f in f_cat:
        dummy = pd.get_dummies(tr_te[f].astype('category'))
        tmp = csr_matrix(dummy)
        sparse_data.append(tmp)
    
    f_num = [f for f in tr_te.columns if 'cont' in f]
    scaler = StandardScaler()
    tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
    sparse_data.append(tmp)
    
    del(tr_te, train, test)
    
    ## sparse train and test data
    xtr_te = hstack(sparse_data, format = 'csr')
    xtrain = xtr_te[:ntrain, :]
    xtest = xtr_te[ntrain:, :]
    
    print('Dim train', xtrain.shape)
    print('Dim test', xtest.shape)
    
    del(xtr_te, sparse_data, tmp)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.isdir(output_path): os.makedirs(output_path) 
    
    ## neural net
    def nn_model():
        model = Sequential()
        model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(200, init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(loss = 'mae', optimizer = 'adadelta')
        return(model)
    
    kf = KFold(n_splits = n_folds, shuffle = True, random_state = 2016)
    
    ## train models
    i = 0
    nbags = 10
    nepochs = 55
    pred_oob = np.zeros(xtrain.shape[0])
    pred_test = np.zeros(xtest.shape[0])
    
    for (inTr, inTe) in kf.split(xtrain, y):
        xtr, ytr, xte, yte = xtrain[inTr], y[inTr], xtrain[inTe], y[inTe]
        pred = np.zeros(xte.shape[0])
        for j in range(nbags):
            model = nn_model()
            fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                      nb_epoch = nepochs,
                                      samples_per_epoch = xtr.shape[0],
                                      verbose = 0)
            if save_oob_predictions: 
                pred += model.predict_generator(generator = batch_generatorp(xte, 800, False),
                                            val_samples = xte.shape[0])[:,0]
            if save_test_predictions:
                pred_test += model.predict_generator(generator = batch_generatorp(xtest, 800, False),
                                                 val_samples = xtest.shape[0])[:,0]
        pred /= nbags
        pred_oob[inTe] = pred
        score = evalerror(yte, pred)
        i += 1
        print('Fold ', i, '- MAE:', score)
    
    if save_oob_predictions:
        maeScore = evalerror(y, pred_oob)
        pd.DataFrame({
            'id': id_train, 
            'loss': np.exp(pred_oob)-featureparams['shift']
            }).to_csv(
            '{}_preds_oob.csv'.format(time), index = False)
    else:
        maeScore = 42.
    print('Total - MAE:', maeScore)
    if save_test_predictions:
        pred_test /= (n_folds*nbags)
        pd.DataFrame({
            'id': id_test, 
            'loss': np.exp(pred_test)-featureparams['shift']
            }).to_csv(
                '{}_submission_keras.csv'.format(time), index = False)
    return maeScore
    
if __name__ == '__main__':
    print sys.argv
    if len(sys.argv) > 1 and sys.argv[1] == 'stam':
        ex.observers.append(MongoObserver.create(url='login1.stampede.tacc.utexas.edu:27017',db_name = "allstate"))
    else:
        ex.observers.append(MongoObserver.create(db_name = "allstate"))
    run = ex.run()  

