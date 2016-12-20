import colored_traceback.always
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
import sys, socket, os
import numpy as np
def xgb_objective(clfparams):
    config={}
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
    # python runModel/tune_xgb.py $SLURM_NODELIST stam
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
        trials = MongoTrials('mongo://login1:27017/allstate/jobs',
                             exp_key='xgb_hyperopt_learning') 
    else:
        trials = Trials()
    
    # Define search space:  
    # xgb_hyperopt_tree_related
    ''' 
    --> 14 searches: for 'learning_rate': 0.001,  best_iter = 26279
    {u'gamma': 1.2146048013139799, u'max_depth': 12, u'min_child_weight': 6}
    {'loss': 1126.8392648025169} --> LB: 1111.93297 (1 hour for test)
    --> 144 searches: for 'learning_rate': 0.01, best_iter = 4649
    {u'gamma': 1.7643768807027944, u'max_depth': 13, u'min_child_weight': 3}
    {'loss': 1127.4636230000001} --> LB: 1112.52785 (3 hours for valid+test)
    {'reg_alpha': 1, 'colsample_bytree': 0.5, 'silent': True, 'colsample_bylevel': 1, 'scale_pos_weight': 1, 'learning_rate': 0.01, 'missing': None, 'max_delta_step': 0, 'nthread': 16, 'base_score': 0.5, 'n_estimators': 5113, 'subsample': 0.8, 'reg_lambda': 1, 'seed': 2016, 'min_child_weight': 3, 'objective': u'reg:linear', 'max_depth': 13, 'gamma': 1.7643768807027944}
    '''  
#     space = {
#         'max_depth':hp.quniform('max_depth',6,14,1),
#         'gamma':hp.uniform('gamma',0,10),
#         'min_child_weight':hp.quniform('min_child_weight',1,10,1)
#         }
 
    # xgb_hyperopt_randomness
    '''
    --> 93 searches
    {u'colsample_bytree': 0.21332019669055208, u'subsample': 0.8709910668393901}
    {'loss': 1125.115356}
    --> 241 searches
    {u'colsample_bytree': 0.2046237651453788, u'subsample': 0.9243246639223455}
    {'loss': 1124.569092}
    --> 311 searches
    {u'colsample_bytree': 0.2189190586811478, u'subsample': 0.9707648911815513}
    {'loss': 1124.550293, 'best_iter': 8108} --> LB: 1111.88553 (11 min for test)
    '''
#     space = {
#         u'gamma': 1.7643768807027944, u'max_depth': 13, u'min_child_weight': 3,
#         'subsample':hp.uniform('subsample',0.05,1.0),
#         'colsample_bytree':hp.uniform('colsample_bytree',0.05,1.0)
#         }
    # xgb_hyperopt_learning
    '''
    --> 11 searches
    {u'reg_alpha': 1.1439554342181042, u'reg_lambda': 2.5778016745822776}
    {'loss': 1124.1636960000001, 'best_iter': 10297}
    --> 500 searches 
    {u'reg_alpha': 1.4561131932721987, u'reg_lambda': 2.7150164332875866}
    {'loss': 1123.626221, 'best_iter': 13416} --> LB: 1111.27926
    {u'reg_alpha': 1.4561131932721987, u'colsample_bytree': 0.2189190586811478, u'silent': True, u'colsample_bylevel': 1, u'scale_pos_weight': 1, u'learning_rate': 0.01, u'missing': None, u'max_delta_step': 0, u'nthread': 16, u'base_score': 0.5, u'n_estimators': 1000000, u'subsample': 0.9707648911815513, u'reg_lambda': 2.7150164332875866, u'seed': 2016, u'min_child_weight': 3, u'objective': u'reg:linear', u'max_depth': 13, u'gamma': 1.7643768807027944}
    '''
    space = {
        u'gamma': 1.7643768807027944, u'max_depth': 13, u'min_child_weight': 3,
        u'colsample_bytree': 0.2189190586811478, u'subsample': 0.9707648911815513,
        'reg_lambda':hp.loguniform('reg_lambda',np.log(1.),np.log(10.)),
        'reg_alpha':hp.loguniform('reg_alpha',np.log(1.),np.log(10.))
        } 
    best = fmin(fn=xgb_objective,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=500)
    print(best)

# col.remove({'exp_key':exp_key})
# col.remove({'exp_key':exp_key,'result.status':'new'})
'''
from pymongo import MongoClient
col = MongoClient('login1').allstate.jobs

import pprint as pp
import numpy as np
exp_key = 'xgb_hyperopt_learning'
experiments = []
loss = []
colCount = col.find({'exp_key':exp_key}).count()
for i in range(0,colCount):
    tmp = col.find({'exp_key':exp_key})[i]
    if tmp['result']['status'] == 'ok' and tmp['result'].has_key('best_iter'):
        result = tmp['result']
#         print  tmp['misc']['vals'].items(),  \
#         "maeScore = {:.4f}, best_iter = {:.4f} "\
#         .format(result['loss'], result['best_iter'])
#         pp.pprint(result['config'])
        experiments.append(tmp)
        loss.append(tmp['result']['loss'])
loss = np.array(loss)
argmin = loss.argmin()
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
'''

#     # kill the servers we started
#     kill_arg = 'killall -r hyperopt-mongo-*'
#     for node in nodes:
#         os.spawnv(os.P_WAIT,'/usr/bin/ssh',["",node,kill_arg])
    
