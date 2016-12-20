import colored_traceback.always
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
import sys, socket, os
import numpy as np
def rf_objective(clfparams):
    config={}
    import datetime
    import sys
    sys.path = [''] + sys.path
    from runModel.rf_run import ex as rfExp
    for (key,value) in clfparams.items():
        config["clfparams.{}".format(key)] = value
    print config
    run = rfExp.run(config_updates=config)
    return {
            'status': 'ok', 
            'loss':run.result,
            'config':run.config
            }

if __name__ == "__main__":
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
    #map(lambda x: x+'.stampede.tacc.utexas.edu', nodes)
    print nodes
    export_arg = 'cd ' + os.getcwd() + '; echo $PWD; hostname; '
    #print export_arg
    server_arg = 'hyperopt-mongo-worker  --mongo=login1:27017/allstateRF' #--poll-interval=0.1'#; < /dev/null > /dev/null &'

    for node in nodes:
        print node, 'is spawning'
        os.spawnv(os.P_NOWAIT,'/usr/bin/ssh',['', node, export_arg, server_arg])
    print 'spawn is finished'
    # we used screen session to launch a mongod at login1
    
    print sys.argv
    if len(sys.argv) > 2 and sys.argv[2] == 'stam':
        trials = MongoTrials('mongo://login1:27017/allstateRF/jobs',
                             exp_key='rf_hyperopt')
    else:
        trials = Trials()
    
    # rf_hyperopt_max_features
    # 100 searches:
#     {u'max_features': 0.18454875562290368}
#     {'loss': 1200.2298096617315} --> LB:
#     space = {
#             'max_features':hp.uniform('max_features',0.05,1.0)
#             }
    '''
    {'searches': 1000}
    {'loss': 1184.2844629110523}
    {u'warm_start': False, u'oob_score': False, u'n_jobs': -1, u'verbose': 0, u'max_leaf_nodes': None, u'bootstrap': True, u'min_samples_leaf': 1.8917159364546094e-05, u'n_estimators': 200, u'min_samples_split': 1.7879145976907858e-05, u'min_weight_fraction_leaf': 0.0, u'criterion': u'mse', u'random_state': 2016, u'min_impurity_split': 1e-07, u'max_features': 0.45447023717474755, u'max_depth': 70}
    '''
    # rf_hyperopt
    space = {
            'max_features':hp.uniform('max_features',0.05,1.0),
            'max_depth':hp.quniform('max_depth',10,100,1),
            'min_samples_leaf':hp.uniform('min_samples_leaf',0.0,0.01),
            'min_samples_split':hp.uniform('min_samples_split',0.0,0.01)
            } 
    best = fmin(fn=rf_objective,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=1000)
    print(best)
# col.remove({'exp_key':exp_key})
# col.remove({'exp_key':exp_key,'result.status':'new'})


'''
from pymongo import MongoClient
col = MongoClient('login1').allstateRF.jobs
exp_key = 'rf_hyperopt'
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
print {'searches':len(experiments)}
print {'loss':loss[argmin]}
print experiments[argmin]['result']['config']['clfparams']
    
'''

#     # kill the servers we started
#     kill_arg = 'killall -r hyperopt-mongo-*'
#     for node in nodes:
#         os.spawnv(os.P_WAIT,'/usr/bin/ssh',["",node,kill_arg])
    
