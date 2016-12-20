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
from xgboost import XGBRegressor as algo
ex = Experiment('xgbRun')
@ex.config
def my_config():
    output_path = "XGBRegressor"
    datadir = 'input'
    n_folds = 10
    clfparams = {'base_score': 0.5,
         'colsample_bylevel': 1,
         'colsample_bytree': 0.5,
         'gamma': 1,
         'learning_rate': 0.01, 
         'max_delta_step': 0,
         'max_depth': 12,
         'min_child_weight': 1,
         'missing': None,
         'n_estimators': 1000000,
         'objective': 'reg:linear',
         'reg_alpha': 1,
         'reg_lambda': 1,
         'scale_pos_weight': 1,
         'seed': 2016,
         'silent': True,
         'subsample': 0.8,
         'nthread': 16}
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
    def evalerror(preds, dtrain):
        labels = dtrain.get_label()
        return 'mae', mean_absolute_error(
            np.exp(preds), np.exp(labels))
    def logregobj(preds, ytrue):
        con = 2
        x =preds-ytrue
        grad =con*x / (np.abs(x)+con)
        hess =con**2 / (np.abs(x)+con)**2
        return grad, hess 
    def logcoshobj(preds, ytrue):
            grad = np.tanh(preds - ytrue)
            hess = 1.0 - grad*grad
            return grad, hess
    
    data = AllStateData(datadir = datadir,
                        include = include,
                        exclude = exclude,
                        **featureparams)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.isdir(output_path): os.makedirs(output_path)  
    xgbIntList = ['max_depth',
                  'n_estimators',
                  'min_child_weight',
                  'max_delta_step']
    for (key,value) in clfparams.items():
        if key in xgbIntList: clfparams[key] = int(value)  
#     clfparams['objective'] = logcoshobj
    alg = algo(**clfparams)
    
    if skip_cross_validation:
        maeScore = 42.
        _run.info['best_iter'] = clfparams['n_estimators']
        _run.info['trainHistory'] = []
        _run.info['validHistory'] = []
    else:
        y = data.ytr
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=2016)
        if save_oob_predictions: pred = pd.DataFrame(-42.0, index = y.index, columns = ['loss'])
        for ikf, (itrain, itest) in enumerate(kf.split(data.Xtr, y)):
            Xtr, ytr, Xte, yte = data.get_train_test_features(itrain, itest)
            alg.fit(Xtr,
                    ytr,
                    eval_set=[(Xtr, ytr), (Xte, yte)], 
                    early_stopping_rounds=300,
                    eval_metric=evalerror,
                    verbose=500)
            if save_oob_predictions:
                pred.iloc[itest] = alg.predict(Xte).reshape(-1,1)
#             maeScoreTrain = data.mae(ytr, alg.predict(Xtr))
#             maeScore = data.mae(yte, pred.iloc[itest].values)
#             for (key,value) in clfparams.items():
#                 print key, value, 
#             print("trainMAE = {:.4f}, oobMAE = {:.4f}, \
#                     best_iter = {:.4f}, best_score = {:.4f}"
#                   .format(maeScoreTrain, maeScore, 
#                           alg.best_iteration, alg.best_score))
            else:
                break # not using cv for parameter tuning for now! TODO: future work.
        maeScore = alg.best_score
        _run.info['best_iter'] = alg.best_iteration
        _run.info['trainHistory'] = alg.evals_result_['validation_0']['mae']
        _run.info['validHistory'] = alg.evals_result_['validation_1']['mae']
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
        alg.fit(Xtr,
                ytr)
        predtest = pd.DataFrame(
            data.inverseScaleTransform(alg.predict(Xte)),
            index = data.testids, columns = ['loss'])
        predtest.to_csv(os.path.join(output_path, filename), index_label='id')
    return maeScore

if __name__ == '__main__':
    print sys.argv
    if len(sys.argv) > 1 and sys.argv[1] == 'stam':
        ex.observers.append(MongoObserver.create(url='login1:27017',db_name = "allstate"))
    else:
        ex.observers.append(MongoObserver.create(db_name = "allstate"))
    run = ex.run()  

