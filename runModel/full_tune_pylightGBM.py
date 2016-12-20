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
    from runModel.pylightGBM_run import ex as xgbExp
    for (key,value) in clfparams.items():
        config["clfparams.{}".format(key)] = value
    print config
    run = xgbExp.run(config_updates=config)
    return {
            'status': 'ok', 
            'loss':run.result,
            'best_iter':run.info['best_iter'],
            'config':run.config
            }

if __name__ == "__main__":
    # in slurm batch script:
    # python runModel/full_tune_pylightGBM.py $SLURM_NODELIST stam
    import colored_traceback.always
    nodelist = sys.argv[1]
    # nodelist = 'c528-404,c549-[001,003-004,006-010,100]'
    # nodelist = 'c528-404,c401-[101,704],c405-[604,701,809,606-608],c543-[006-010]'
    nodes = []
    for h in nodelist.strip().split('c'):
        if h == '':
            continue
        if h.find('[') == -1:
            nodes.append('c'+h.replace(',',''))
        else:
            hosts = h.strip(',').replace(']','').split('[')
            cStart = 'c'+hosts[0]
            for cEnd in hosts[1].split(','):
                if cEnd.find('-') == -1:
                    nodes.append(cStart+cEnd)
                else:
                    s1,e1 = cEnd.split('-')
                    for i in range(int(s1),int(e1)+1):
                        nodes.append(cStart+str(i).zfill(3))
    print nodes
    export_arg = 'cd ' + os.getcwd() + '; echo $PWD; hostname; '
    #print export_arg
    server_arg = 'hyperopt-mongo-worker  --mongo=login1:27017/allstate' #--poll-interval=0.1'#; < /dev/null > /dev/null &'

    for node in nodes:
        print node, 'is spawning'
        os.spawnv(os.P_NOWAIT,'/usr/bin/ssh',['', node, export_arg, server_arg])
    print 'spawn is finished'

    print sys.argv
    if len(sys.argv) > 2 and sys.argv[2] == 'stam':
    # we used screen session to launch a mongod at login1
        trials = MongoTrials('mongo://login1:27017/allstate/jobs', 
                             exp_key='full_xgb_hyperopt_pylightGBM')
    else:
        trials = Trials()
    space = {
             'max_bin':hp.quniform('max_bin',50,500,1),
             'min_data_in_leaf':hp.quniform('min_data_in_leaf',50,500,1),
             'num_leaves':hp.quniform('num_leaves',50,500,1),
             'feature_fraction':hp.uniform('feature_fraction',0.05,1.0),
             'bagging_fraction':hp.uniform('bagging_fraction',0.05,1.0),
             'metric':hp.choice('metric',['l1','l2'])
             }  
    
    best = fmin(fn=xgb_objective,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=2000)
    print(best)
    '''
exp_key='full_xgb_hyperopt_pylightGBM'
col = MongoClient('login1').allstate.jobs
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
print {'loss':loss[argmin], 'best_iter':experiments[argmin]['result']['best_iter']}
print experiments[argmin]['result']['config'][u'clfparams']

alg.set_params(**space)
    '''