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
from pylightgbm.models import GBMRegressor as algo
ex = Experiment('pylightgbmRun')
@ex.config
def my_config():
    output_path = "pylightgbmRun"
    datadir = 'input'
    n_folds = 10
    os.environ['LIGHTGBM_EXEC'] = "/home/frank/Downloads/LightGBM/lightgbm"
    clfparams = {'config': '',
         'exec_path': '/home/frank/Downloads/LightGBM/lightgbm',
         'application': 'regression',
          'bagging_fraction': 1.0,
          'bagging_freq': 10,
          'bagging_seed': 3,
          'boosting_type': 'gbdt',
          'early_stopping_round': 300,
          'feature_fraction': 1.0,
          'feature_fraction_seed': 2,
          'is_training_metric': False,
          'is_unbalance': False,
          'learning_rate': 0.01,
          'max_bin': 255,
          'metric': 'l2',# , l1
          'metric_freq': 100,
          'min_data_in_leaf': 10,
          'num_class': 1,
          'num_iterations': 1000000,
          'num_leaves': 127,
          'num_threads': 16,
          'tree_learner': 'serial',
          'min_sum_hessian_in_leaf':10.0,
         'verbose': 1}
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
    def evalerror(preds, labels):
        return mean_absolute_error(
            np.exp(preds), np.exp(labels))
    
    data = AllStateData(datadir = datadir,
                        include = include,
                        exclude = exclude,
                        **featureparams)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.isdir(output_path): os.makedirs(output_path)  
    xgbIntList = ['num_leaves',
                  'num_iterations',
                  'min_data_in_leaf',
                  'max_bin',
                  'max_position']
    for (key,value) in clfparams.items():
        if key in xgbIntList: clfparams[key] = int(value)  
        
    alg = algo(**clfparams)
    
    if skip_cross_validation:
        maeScore = 42.
        _run.info['best_iter'] = clfparams['num_iterations']
    else:
        y = data.ytr
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=2016)
        if save_oob_predictions: pred = pd.DataFrame(-42.0, index = y.index, columns = ['loss'])
        for ikf, (itrain, itest) in enumerate(kf.split(data.Xtr, y)):
            Xtr, ytr, Xte, yte = data.get_train_test_features(itrain, itest)
            alg.fit(Xtr,
                    ytr,
                    test_data = [(Xte, yte)])
            oob = alg.predict(Xte).reshape(-1,1)
            if save_oob_predictions:
                pred.iloc[itest] = oob
            else:
                maeScore = evalerror(oob, yte.values)
                break # not using cv for parameter tuning for now! TODO: future work.
        if save_oob_predictions: maeScore = evalerror(pred.values, y.values)
        _run.info['best_iter'] = alg.best_round
        print alg.best_round
    # Optionally save oob predictions
        if save_oob_predictions:
            filename = '{}_oob_pred_lightGBM.csv'.format(time)
            pd.DataFrame(data.inverseScaleTransform(pred.values),
                         index=data.trainids,columns=['loss']).to_csv(
                             os.path.join(output_path, filename),
                         index_label='id')
    # Optionally generate test predictions
    if save_test_predictions:
        filename = '{}_test_pred_lightGBM.csv'.format(time)
        Xtr, ytr, Xte, _ = data.get_train_test_features() 
        for k in alg.get_params().keys(): 
            if k != 'model':
                print k, alg.get_params()[k]
        alg.fit(Xtr, ytr)
        predtest = pd.DataFrame(
            data.inverseScaleTransform(alg.predict(Xte)),
            index = data.testids, columns = ['loss'])
        predtest.to_csv(os.path.join(output_path, filename), index_label='id')
    return maeScore

if __name__ == '__main__':
    print sys.argv
    if len(sys.argv) > 1 and sys.argv[1] == 'stam':
        ex.observers.append(MongoObserver.create(url='login1.stampede.tacc.utexas.edu:27017',db_name = "allstate"))
    else:
        ex.observers.append(MongoObserver.create(db_name = "allstate"))
    run = ex.run()  

