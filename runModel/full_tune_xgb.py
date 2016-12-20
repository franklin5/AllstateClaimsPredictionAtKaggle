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
    from runModel.xgb_run import ex as xgbExp
    for (key,value) in clfparams.items():
        config["clfparams.{}".format(key)] = value
    print config
    run = xgbExp.run(config_updates=config)
    return {
            'status': 'ok', 
            'loss':run.result,
            'best_iter':run.info['best_iter'],
            'info':{'trainHistory':run.info['trainHistory'],
                    'validHistory':run.info['validHistory']},
            'config':run.config
            }

if __name__ == "__main__":
    # in slurm batch script:
    # python runModel/full_tune_xgb.py $SLURM_NODELIST stam
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
                             exp_key='full_xgb_hyperopt')
    else:
        trials = Trials()
    space = {#'learning_rate':hp.uniform('learning_rate',0.001,0.1),
             #'n_estimators':hp.quniform('n_estimators',100,3000,1),
             'max_depth':hp.quniform('max_depth',6,14,1),
             'gamma':hp.uniform('gamma',0,4),
             'min_child_weight':hp.quniform('min_child_weight',1,10,1),
             'subsample':hp.uniform('subsample',0.05,1.0),
             'colsample_bytree':hp.uniform('colsample_bytree',0.05,1.0),
             'reg_lambda':hp.loguniform('reg_lambda',np.log(1.),np.log(100.)),
             'reg_alpha':hp.loguniform('reg_alpha',np.log(1.),np.log(100.))}  
    ''' 
    # 46 searches
    {u'colsample_bytree': 0.4238334128807261,
     u'gamma': 1.674820260013389,
     u'max_depth': 9,
     u'min_child_weight': 10,
     u'reg_alpha': 5.0533091800040975,
     u'reg_lambda': 7.354467092857866,
     u'subsample': 0.8906921411883283}
    {'loss': 1127.3790276695147}
    # 336 searches
    {u'colsample_bytree': 0.19802732554358685,
     u'gamma': 0.6644212863403848,
     u'max_depth': 8,
     u'min_child_weight': 6,
     u'reg_alpha': 6.943299183140909,
     u'reg_lambda': 1.317547537758126,
     u'subsample': 0.999237768758001}
    {'loss': 1124.5014650000001, 'best_iter': 8737} --> LB: 1112.95477
    # 487 searches
    {u'colsample_bytree': 0.20019686262655803,
     u'gamma': 0.9514925323948685,
     u'max_depth': 12,
     u'min_child_weight': 9,
     u'reg_alpha': 3.542156115376599,
     u'reg_lambda': 5.059863811267914,
     u'subsample': 0.9492742393677509}
    {'loss': 1123.8199460000001, 'best_iter': 7023}
    # 793 searches
    {u'colsample_bytree': 0.20115723827659251,
     u'gamma': 1.1997673613959643,
     u'max_depth': 11,
     u'min_child_weight': 9,
     u'reg_alpha': 5.776300110829684,
     u'reg_lambda': 7.097707130630405,
     u'subsample': 0.9599636961269256}
    {'loss': 1123.6248780000001, 'best_iter': 9930}
    # 892 searches
    {u'colsample_bytree': 0.21794243714071115,
     u'gamma': 1.054799522216147,
     u'max_depth': 12,
     u'min_child_weight': 3,
     u'reg_alpha': 2.796173535473323,
     u'reg_lambda': 3.915744575886336,
     u'subsample': 0.9821429529742869}
    {'loss': 1123.5485839999999, 'best_iter': 5530}
    # 3026 searches
    {'loss': 1123.1872559999999, 'best_iter': 7771}
    {u'reg_alpha': 6.928178111723366, u'colsample_bytree': 0.20660778240312197, u'silent': True, u'colsample_bylevel': 1, u'scale_pos_weight': 1, u'learning_rate': 0.01, u'missing': None, u'max_delta_step': 0, u'nthread': 16, u'base_score': 0.5, u'n_estimators': 1000000, u'subsample': 0.9804855905758271, u'reg_lambda': 4.471792660171384, u'seed': 2016, u'min_child_weight': 2, u'objective': u'reg:linear', u'max_depth': 12, u'gamma': 0.9789220494518855}
    '''
    best = fmin(fn=xgb_objective,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=2000)
    print(best)
    # kill the servers we started
    #kill_arg = 'killall -r hyperopt-mongo-*'
    #for node in nodes:
    #    os.spawnv(os.P_WAIT,'/usr/bin/ssh',["",node,kill_arg])
    '''
exp_key='full_xgb_hyperopt'
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