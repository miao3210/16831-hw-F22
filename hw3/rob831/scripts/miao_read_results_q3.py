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
    
    '''folders = ['../../data/q3_hparam1_LunarLander-v3_31-10-2022_19-41-27', 
        '../../data/q3_hparam2_LunarLander-v3_31-10-2022_19-43-11', 
        '../../data/q3_hparam3_LunarLander-v3_31-10-2022_19-44-30']'''
    '''folders = ['../../data/q3_hparam1_LunarLander-v3_30-10-2022_17-46-26', 
        #'../../data/q3_hparam2_LunarLander-v3_30-10-2022_17-48-02', 
        '../../data/q3_hparam3_LunarLander-v3_30-10-2022_17-49-41', 
        '../../data/q3_hparam4_LunarLander-v3_31-10-2022_20-40-37']'''
    folders = [
        'q3_hparam0_target_update_freq_3000_original_LunarLander-v3_01-11-2022_18-59-16',
        'q3_hparam1_target_update_freq_1000_LunarLander-v3_01-11-2022_18-59-38',
        'q3_hparam2_target_update_freq_300_LunarLander-v3_01-11-2022_19-00-02',
        'q3_hparam3_target_update_freq_10000_LunarLander-v3_01-11-2022_19-00-24'
    ]
    folders = ['../../data/'+f for f in folders]
    print(folders)
    logs = {} # dict_keys(['doubledqn_1', 'doubledqn_2', 'doubledqn_3', 'dqn_1', 'dqn_2', 'dqn_3'])
    for folder in folders:
        logdir = os.path.join(folder, 'events*')
        print('logdir={}'.format(logdir))
        eventfile = glob.glob(logdir)[0]

        X, Y, Z, stepX, stepY, stepZ = get_section_results(eventfile)
        logs[folder.split('/q3_')[-1].split('_Lunar')[0]] = [X, stepX, Y, stepY, Z, stepZ]
        #for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        #    pass #print('Iteration {:d} | Train steps: {:d} | Return: {} | Best Return: {}'.format(i, int(x), y, z))
    #pdb.set_trace()
    #assert len(logs['hparam1'][3]) == len(logs['hparam4'][3])
    #assert len(logs['hparam3'][3]) == len(logs['hparam4'][3])

    plt.figure(figsize=(10,4))
    for k in logs.keys():
        plt.plot(logs[k][3], logs[k][2], label=k)
    #plt.plot(stepY, logs['hparam1'][2], label='hparam1') #'lr=0.1') #'original')
    #plt.plot(stepY, logs['hparam4'][2], label='hparam4') #'lr=0.01') #'larger size')
    #plt.plot(stepY, logs['hparam3'][2], label='hparam3') #'lr=0.001') #'largest size')
    plt.xlabel('time step')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend(loc = 'lower left')
    plt.ylabel('average reward ')
    plt.title('q3_LunarLander-v3 different model size: both layer size and layer number')
    plt.savefig('./figures/q3.png', bbox_inches='tight')