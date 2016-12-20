import colored_traceback.always
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
import sys, socket, os
import numpy as np
def xgb_objective(clfparams):
    config={}
    import datetime
    import sys
    sys.path = [''] + sys.path
    from runModel.knn_run import ex as xgbExp
    for (key,value) in clfparams.items():
        config["clfparams.{}".format(key)] = value
    print config
    run = xgbExp.run(config_updates=config)
    return {
            'status': 'ok', 
            'loss':run.result,
            'config':run.config
            }

if __name__ == "__main__":
#     trials = Trials()
    trials = MongoTrials('mongo://localhost:27017/allstate/jobs', 
                             exp_key='full_knn_hyperopt')
    space = {'n_neighbors': hp.quniform('n_neighbors',4,100,1),
             'weights': hp.choice('weights',['uniform','distance'])
             }
    best = fmin(fn=xgb_objective,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=50)
    print(best)
    # kill the servers we started
    #kill_arg = 'killall -r hyperopt-mongo-*'
    #for node in nodes:
    #    os.spawnv(os.P_WAIT,'/usr/bin/ssh',["",node,kill_arg])
    '''
exp_key='full_knn_hyperopt'
col = MongoClient().allstate.jobs
colCount = col.find({'exp_key':exp_key}).count()
experiments = []
loss = []
for i in range(0,colCount):
    tmp = col.find({'exp_key':exp_key})[i]
    if tmp['result']['status'] == 'ok':
        experiments.append(tmp)
        loss.append(tmp['result']['loss'])
import numpy as np
loss = np.array(loss)
argmin = loss.argmin()
import pprint as pp
space = experiments[argmin]['misc']['vals']
xgbIntList = ['max_depth', 'n_estimators', 'min_child_weight', 'max_delta_step']
for k in space.keys(): 
    if k in xgbIntList:
        space[k] = int(space[k][0])
    else:
        space[k] = space[k][0]
pp.pprint(space)
print {'number of searches':len(experiments)}
print {'loss':loss[argmin]}
print experiments[argmin]['result']['config'][u'clfparams']
    '''