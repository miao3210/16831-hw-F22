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
        pdb.set_trace()
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
        #if len(X) > 120:tensor
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
    
    folders = ['../../data/q4_100_1_CartPole-v0_31-10-2022_02-07-06', 
        '../../data/q4_10_10_CartPole-v0_31-10-2022_02-07-06', 
        '../../data/q4_1_100_CartPole-v0_31-10-2022_02-07-06', 
        '../../data/q4_ac_1_1_CartPole-v0_31-10-2022_02-01-07']
    
    logs = {} # dict_keys(['doubledqn_1', 'doubledqn_2', 'doubledqn_3', 'dqn_1', 'dqn_2', 'dqn_3'])
    for folder in folders:
        logdir = os.path.join(folder, 'events*')
        print('logdir={}'.format(logdir))
        eventfile = glob.glob(logdir)[0]

        X, Y, Z, stepX, stepY, stepZ = get_section_results(eventfile)
        logs[folder.split('/q4_')[-1].split('_Cart')[0]] = [X, stepX, Y, stepY, Z, stepZ]
        #for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        #    pass #print('Iteration {:d} | Train steps: {:d} | Return: {} | Best Return: {}'.format(i, int(x), y, z))
    #pdb.set_trace()
    assert len(logs['ac_1_1'][3]) == len(logs['1_100'][3])
    assert len(logs['1_100'][3]) == len(logs['10_10'][3])
    assert len(logs['100_1'][3]) == len(logs['10_10'][3])

    plt.figure(figsize=(10,4))
    for k in logs.keys():
        plt.plot(logs[k][3], logs[k][2], label='ntu=' + str(k.split('_')[-2]) + ' ngsptu=' + str(k.split('_')[-1]))
    plt.xlabel('time step')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend(loc = 'upper left')
    plt.ylabel('average reward ')
    plt.title('q4_CartPole-v0 different number of target updates and gradient updates')
    plt.savefig('./figures/q4.png', bbox_inches='tight')