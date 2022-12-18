import argparse
import glob
import os
import tensorflow.compat.v1 as tf

## miao
import pdb 
import matplotlib.pyplot as plt 
import numpy as np

tf.disable_v2_behavior()

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    stepX = []
    stepY = []
    stepZ = []
    for e in tf.train.summary_iterator(file):
        #pdb.set_trace()
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
                stepX.append(e.step)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
                stepY.append(e.step)
            elif v.tag == 'Train_BestReturn':
                Z.append(v.simple_value)
                stepZ.append(e.step)
        #if len(X) > 120:
        #    break
    return X, Y, Z, stepX, stepY, stepZ

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    '''
    ../../data/q2
    '''
    #args = parser.parse_args()

    #logdir = os.path.join(args.logdir, 'events*')
    
    foldersdqn = ['../../data/q2_dqn_1_LunarLander-v3_30-10-2022_17-16-04', ]*1 + \
        ['../../data/q2_dqn_2_LunarLander-v3_30-10-2022_17-15-54', ]*1 + \
        ['../../data/q2_dqn_3_LunarLander-v3_30-10-2022_17-16-01']*1
    
    #foldersdoubledqn = [
        #'q2_doubledqn_1_LunarLander-v3_30-10-2022_17-26-12', 
        #'q2_doubledqn_2_LunarLander-v3_30-10-2022_17-32-18', 
        #'q2_doubledqn_3_LunarLander-v3_30-10-2022_17-32-11'
    #    ] #'''

    #foldersdoubledqn = [
        #'q2_doubledqn_1_LunarLander-v3_01-11-2022_21-03-29', #'q2_doubledqn_1_LunarLander-v3_01-11-2022_18-56-13',
        #'q2_doubledqn_2_LunarLander-v3_01-11-2022_21-03-29', #'q2_doubledqn_2_LunarLander-v3_01-11-2022_18-56-13',
        #'q2_doubledqn_3_LunarLander-v3_01-11-2022_21-03-29', #'q2_doubledqn_3_LunarLander-v3_01-11-2022_18-56-13'
    #    ] # haohong
    
    #foldersdoubledqn = [
        #'q2_doubledqn_4_LunarLander-v3_01-11-2022_21-22-30',
        #'q2_doubledqn_5_LunarLander-v3_01-11-2022_21-22-30',
        #'q2_doubledqn_6_LunarLander-v3_01-11-2022_21-22-30'
    #]

    foldersdoubledqn = [
        #'q2_doubledqn_9_LunarLander-v3_01-11-2022_21-44-20',
        'q2_doubledqn_2_LunarLander-v3_01-11-2022_21-44-20', # 8
        'q2_doubledqn_1_LunarLander-v3_01-11-2022_21-44-20', # 7
        'q2_doubledqn_3_LunarLander-v3_01-11-2022_21-46-40' # 12
        ]

    #foldersdoubledqn = [
        #'q2_doubledqn_10_LunarLander-v3_01-11-2022_21-46-40',
        #'q2_doubledqn_11_LunarLander-v3_01-11-2022_21-46-40',
        #'q2_doubledqn_12_LunarLander-v3_01-11-2022_21-46-40'
    #]
    foldersdoubledqn = ['../../data/'+f for f in foldersdoubledqn]
    
    
    folders = foldersdqn + foldersdoubledqn
    logs = {} # dict_keys(['doubledqn_1', 'doubledqn_2', 'doubledqn_3', 'dqn_1', 'dqn_2', 'dqn_3'])
    for folder in folders:
        logdir = os.path.join(folder, 'events*')
        print('logdir={}'.format(logdir))
        eventfile = glob.glob(logdir)[0]

        X, Y, Z, stepX, stepY, stepZ = get_section_results(eventfile)
        logs[folder.split('/q2_')[-1].split('_Lunar')[0]] = [X, stepX, Y, stepY, Z, stepZ]
        #for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        #    pass #print('Iteration {:d} | Train steps: {:d} | Return: {} | Best Return: {}'.format(i, int(x), y, z))
    #pdb.set_trace()
    '''
    assert len(logs['doubledqn_1'][3]) == len(logs['doubledqn_2'][3])
    assert len(logs['doubledqn_3'][3]) == len(logs['doubledqn_2'][3])
    assert len(logs['dqn_1'][3]) == len(logs['dqn_2'][3])
    assert len(logs['dqn_3'][3]) == len(logs['dqn_2'][3])
    assert len(logs['doubledqn_1'][5]) == len(logs['doubledqn_2'][5])
    assert len(logs['doubledqn_3'][5]) == len(logs['doubledqn_2'][5])
    assert len(logs['dqn_1'][5]) == len(logs['dqn_2'][5])
    assert len(logs['dqn_3'][5]) == len(logs['dqn_2'][5])
    '''
    aver_dqn = np.array([logs['dqn_1'][2], logs['dqn_2'][2], logs['dqn_3'][2]]).mean(axis=0)
    print(aver_dqn.shape)

    doubeldqn = []
    for k in logs.keys():
        if 'double' in k:
            doubeldqn.append([logs[k][2], logs[k][3]]) 
    aver_ddqn = np.array(doubeldqn)[:, 0].mean(axis=0)
    print(aver_ddqn.shape)
    '''dqn = []
    for k in logs.keys():
        if 'double' in k:
            dqn.append([logs[k][2], logs[k][3]]) 
    aver_dqn = np.array(dqn)[:, 0].mean(axis=0)'''

    plt.figure(figsize=(10,4))
    plt.plot(stepY, aver_dqn, label='dqn')
    plt.plot(stepY, aver_ddqn, label='double dqn')
    plt.xlabel('time step')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend(loc = 'upper left')
    plt.ylabel('average reward ')
    plt.ylim([-260, 155])
    plt.title('q2_LunarLander-v3 dqn and double dqn')
    plt.savefig('./figures/q2.png', bbox_inches='tight')