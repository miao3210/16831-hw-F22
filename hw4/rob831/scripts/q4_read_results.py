import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt 

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    stepX = []
    stepY = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_AverageReturn':
                X.append(v.simple_value)
                stepX.append(e.step)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
                stepY.append(e.step)
    return X, Y, stepX, stepY

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    #args = parser.parse_args()

    paths = ['hw4_q4_reacher_ensemble1_reacher-rob831-v0_15-11-2022_19-26-40/',
        'hw4_q4_reacher_ensemble3_reacher-rob831-v0_15-11-2022_19-30-59/',
        'hw4_q4_reacher_ensemble5_reacher-rob831-v0_15-11-2022_19-42-43/',
        'hw4_q4_reacher_horizon15_reacher-rob831-v0_15-11-2022_18-28-23/',
        'hw4_q4_reacher_horizon30_reacher-rob831-v0_15-11-2022_18-44-56/',
        'hw4_q4_reacher_horizon5_reacher-rob831-v0_15-11-2022_17-51-10/',
        'hw4_q4_reacher_numseq1000_reacher-rob831-v0_15-11-2022_19-14-52/',
        'hw4_q4_reacher_numseq100_reacher-rob831-v0_15-11-2022_01-50-03/']
    paths = ['../../data/' + p for p in paths]

    plt.figure()
    for p in paths:
        logdir = os.path.join(p, 'events*')
        print(logdir)
        eventfile = glob.glob(logdir)[0]

        X, Y, stepX, stepY = get_section_results(eventfile)
        #print(X)
        #print(Y)
        
        for i, (x, y) in enumerate(zip(stepY, Y)):
            print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

        label = p.split('reacher')[1].split('_')[1]
        print(label)
        plt.plot(stepY, Y, label=label)
    plt.legend()
    plt.title('q4: ablation study')
    plt.xlabel('iteration')
    plt.ylabel('eval average reward')
    plt.savefig('./figure/q4.png')
    plt.close()


    for p in paths:
        logdir = os.path.join(p, 'events*')
        print(logdir)
        if not 'ensemble' in p:
            continue
        eventfile = glob.glob(logdir)[0]

        X, Y, stepX, stepY = get_section_results(eventfile)
        #print(X)
        #print(Y)
        
        for i, (x, y) in enumerate(zip(stepY, Y)):
            print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

        label = p.split('reacher')[1].split('_')[1]
        print(label)
        plt.plot(stepY, Y, label=label)
    plt.legend()
    plt.title('q4: ensemble size')
    plt.xlabel('iteration')
    plt.ylabel('eval average reward')
    plt.savefig('./figure/q4_ensemble.png')
    plt.close()


    for p in paths:
        logdir = os.path.join(p, 'events*')
        print(logdir)
        if not 'numseq' in p:
            continue
        eventfile = glob.glob(logdir)[0]

        X, Y, stepX, stepY = get_section_results(eventfile)
        #print(X)
        #print(Y)
        
        for i, (x, y) in enumerate(zip(stepY, Y)):
            print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

        label = p.split('reacher')[1].split('_')[1]
        print(label)
        plt.plot(stepY, Y, label=label)
    plt.legend()
    plt.title('q4: number of candidate action sequences')
    plt.xlabel('iteration')
    plt.ylabel('eval average reward')
    plt.savefig('./figure/q4_numseq.png')
    plt.close()


    for p in paths:
        logdir = os.path.join(p, 'events*')
        print(logdir)
        if not 'horizon' in p:
            continue
        eventfile = glob.glob(logdir)[0]

        X, Y, stepX, stepY = get_section_results(eventfile)
        #print(X)
        #print(Y)
        
        for i, (x, y) in enumerate(zip(stepY, Y)):
            print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

        label = p.split('reacher')[1].split('_')[1]
        print(label)
        plt.plot(stepY, Y, label=label)
    plt.legend()
    plt.title('q4: horizon size')
    plt.xlabel('iteration')
    plt.ylabel('eval average reward')
    plt.savefig('./figure/q4_horizon.png')
    plt.close()
