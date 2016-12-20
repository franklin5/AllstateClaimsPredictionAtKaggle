from __future__ import division
import os
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import sys
sys.path = [''] + sys.path 
from src.allstate_data import AllStateData
from sacred import Experiment
from sacred.observers import MongoObserver
import colored_traceback.always
from sklearn.ensemble import RandomForestRegressor as algo
ex = Experiment('RandomForestRegressorRun')
@ex.config
def my_config():
    output_path = "RandomForestRegressor"
    datadir = 'input'
    n_folds = 10
    clfparams = {
             'bootstrap': True,
             'criterion': 'mse',
             'max_depth': None,
             'max_features': 'auto',
             'max_leaf_nodes': None,
             'min_impurity_split': 1e-07,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'min_weight_fraction_leaf': 0.0,
             'n_estimators': 200,
             'n_jobs': -1,
             'oob_score': False,
             'random_state': 2016,
             'verbose': 0,
             'warm_start': False}
    include = []
    exclude = []
    featureparams = {'shift':200.0}
    save_oob_predictions  = False
    save_test_predictions = False
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
    data = AllStateData(datadir = datadir,
                        include = include,
                        exclude = exclude,
                        **featureparams)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.isdir(output_path): os.makedirs(output_path)
        
    intList = ['max_depth',
              'n_estimators',
              'max_leaf_nodes']
    for (key,value) in clfparams.items():
        if key in intList and value != None:
            clfparams[key] = int(value)

    alg = algo(**clfparams)
    
    if skip_cross_validation:
        maeScore = 42.
    else:
        y = data.ytr
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=2016)
        pred = pd.DataFrame(-42.0, index = y.index, columns = ['loss'])
        for ikf, (itrain, itest) in enumerate(kf.split(data.Xtr, y)):
            Xtr, ytr, Xte, yte = data.get_train_test_features(itrain, itest)
            print alg.get_params()
            alg.fit(Xtr, ytr.values.reshape(-1,))
            pred.iloc[itest] = alg.predict(Xte).reshape(-1,1)
            maeScoreTrain = data.mae(ytr, alg.predict(Xtr))
            maeScore = data.mae(yte, pred.iloc[itest].values)
            for (key,value) in clfparams.items():
                print key, value, 
            print("trainMAE = {:.4f}, \
                  oobMAE = {:.4f}".format(maeScoreTrain, maeScore))
            break
    # Optionally save oob predictions
        if save_oob_predictions:
            filename = '{}_oob_pred.csv'.format(time)
            pd.DataFrame(data.inverseScaleTransform(pred.values),
                         index=data.trainids,columns=['loss']).to_csv(
                             os.path.join(output_path, filename),
                         index_label='id')
    # Optionally generate test predictions
    if save_test_predictions:
        filename = '{}_test_pred.csv'.format(time)
        Xtr, ytr, Xte, _ = data.get_train_test_features() 
        print alg.get_params()
        alg.fit(Xtr, ytr.values.reshape(-1,))
        predtest = pd.DataFrame(
            data.inverseScaleTransform(alg.predict(Xte)),
            index = data.testids, columns = ['loss'])
        predtest.to_csv(os.path.join(output_path, filename), index_label='id')
    return maeScore

if __name__ == '__main__':
    print sys.argv
    if len(sys.argv) > 1 and sys.argv[1] == 'stam':
        ex.observers.append(MongoObserver.create(url='login1:27017',db_name = "allstateRF"))
    else:
        ex.observers.append(MongoObserver.create(db_name = "allstateRF"))
    run = ex.run()  