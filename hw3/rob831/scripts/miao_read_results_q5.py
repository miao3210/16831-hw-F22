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
    
    folders = [#'../../data/q5_10_10_HalfCheetah-v2_31-10-2022_19-04-48',
        #'../../data/q5_10_10_HalfCheetah-v2_31-10-2022_17-30-38', fail
        #'../../data/q5_10_10_mean_HalfCheetah-v2_02-11-2022_14-49-31',
        #'../../data/q5_10_10_seed1_HalfCheetah-v2_02-11-2022_14-19-45', 
        #'../../data/q5_10_10_seed2_HalfCheetah-v2_02-11-2022_14-19-45',
        #'../../data/q5_10_10_seed3_HalfCheetah-v2_02-11-2022_14-19-45',
        #'../../data/q5_10_10_np_HalfCheetah-v2_02-11-2022_16-31-03',
        #'../../data/q5_10_10_both_HalfCheetah-v2_02-11-2022_16-33-10',
        '../../data/q5_10_10_ultimate_HalfCheetah-v2_04-11-2022_12-57-16',
        '../../data/q5_10_10_InvertedPendulum-v2_31-10-2022_17-30-38']
    
    logs = {} # dict_keys(['doubledqn_1', 'doubledqn_2', 'doubledqn_3', 'dqn_1', 'dqn_2', 'dqn_3'])
    for folder in folders:
        logdir = os.path.join(folder, 'events*')
        print('logdir={}'.format(logdir))
        eventfile = glob.glob(logdir)[0]

        X, Y, Z, stepX, stepY, stepZ = get_section_results(eventfile)
        logs[folder.split('/q5_')[-1].split('_31')[0].split('_02')[0]] = [X, stepX, Y, stepY, Z, stepZ]
        #for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        #    pass #print('Iteration {:d} | Train steps: {:d} | Return: {} | Best Return: {}'.format(i, int(x), y, z))
    #pdb.set_trace()

    
    for k in logs.keys():
        plt.figure(figsize=(10,4))
        if 'HalfCheetah' in k:
            #pdb.set_trace()
            plt.plot(logs[k][3], [-40]*len(logs[k][3]), '--', label='-40')
            plt.plot(logs[k][3], [140]*len(logs[k][3]), '--', label='140')
            #plt.plot([150, 150], [-100, 0], '-')
        else:
            plt.plot(stepY, [100]*len(stepY), '--', label='100')
            #plt.plot([100, 100], [0,200], '-')
        if 'HalfCheetah' in k and 'mean' in k:
            plt.plot(logs[k][3], logs[k][2], label='10_10_HalfCheetah-v2') #k)
        else:
            plt.plot(logs[k][3], logs[k][2], label=k)
        plt.xlabel('time step')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.legend(loc = 'lower right')
        plt.ylabel('average reward ')
        plt.title(k)
        plt.savefig('./figures/q5_'+k+'.png', bbox_inches='tight')
    

    # half cheeftah
    plt.figure(figsize=(10,4))
    plt.plot(logs[k][3], [-40]*len(logs[k][3]), '--', label='-40')
    for k in logs.keys():
        if 'HalfCheetah' in k:
            #pdb.set_trace()
            plt.plot(logs[k][3], logs[k][2], label=k)

    plt.xlabel('time step')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend(loc = 'lower right')
    plt.ylabel('average reward ')
    plt.title('10_10_HalfCheetah-v2')
    plt.savefig('./figures/q5_HalfCheetah.png', bbox_inches='tight')