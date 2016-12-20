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
    from runModel.xgb_run_fair import ex as xgbExp
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
    # python runModel/full_tune_xgb_fair_obj.py $SLURM_NODELIST stam
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
                             exp_key='full_xgb_hyperopt_fair_obj')
    else:
        trials = Trials()
    '''
    {'number of searches': 495}
    {'loss': 1121.164673, 'best_iter': 5959} --> LB:1109.39039
    {u'colsample_bytree': 0.2309253706796382, u'silent': True, u'missing': None, u'scale_pos_weight': 1, u'colsample_bylevel': 1, u'max_delta_step': 0, u'nthread': 16, u'base_score': 0.5, u'n_estimators': 1000000, u'subsample': 0.9850550524511695, u'eta': 0.01, u'min_child_weight': 1, u'alpha': 3.88705123516055, u'seed': 2016, u'max_depth': 14, u'gamma': 0.7707693199212378, u'lambda': 16.521895954977534}
    {'number of searches': 1783}
    {'loss': 1120.9517820000001, 'best_iter': 11804}
{u'colsample_bytree': 0.19177873568880957, u'silent': True, u'missing': None, u'scale_pos_weight': 1, u'colsample_bylevel': 1, u'max_delta_step': 0, u'nthread': 16, u'base_score': 0.5, u'n_estimators': 1000000, u'subsample': 0.9999409699274003, u'eta': 0.01, u'min_child_weight': 2, u'alpha': 1.941088877046666, u'seed': 2016, u'max_depth': 17, u'gamma': 1.2175144489387606, u'lambda': 2.7786964200524222}
'''
    space = {
             'max_depth':hp.quniform('max_depth',6,30,1),
             'gamma':hp.uniform('gamma',0,4),
             'min_child_weight':hp.quniform('min_child_weight',1,10,1),
             'subsample':hp.uniform('subsample',0.05,1.0),
             'colsample_bytree':hp.uniform('colsample_bytree',0.05,1.0),
             'lambda':hp.loguniform('lambda',np.log(1.),np.log(100.)),
             'alpha':hp.loguniform('alpha',np.log(1.),np.log(100.))
             }  
    
    best = fmin(fn=xgb_objective,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=3000)
    print(best)
    # kill the servers we started
    #kill_arg = 'killall -r hyperopt-mongo-*'
    #for node in nodes:
    #    os.spawnv(os.P_WAIT,'/usr/bin/ssh',["",node,kill_arg])
    '''
exp_key='full_xgb_hyperopt_fair_obj'
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