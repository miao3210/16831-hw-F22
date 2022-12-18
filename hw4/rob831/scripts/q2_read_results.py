import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

## miao
import pdb 
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
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    #args = parser.parse_args()

    paths = ['hw4_q2_obstacles_singleiteration_obstacles-rob831-v0_14-11-2022_20-17-11']
    paths = ['../../data/' + p for p in paths]

    for p in paths:
        logdir = os.path.join(p, 'events*')
        print(logdir)
        eventfile = glob.glob(logdir)[0]

        X, Y, stepX, stepY = get_section_results(eventfile)
        #print(X)
        #print(Y)
        for i, (x, y) in enumerate(zip(X, Y)):
            print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
        
        #plt.plot(stepX, X, label=label + ' train average reward')
        plt.plot(stepX, X, '*', label='train')
        plt.plot(stepY, Y, '*', label='eval')
        plt.legend()
        plt.title('q2: action selection')
        plt.xlabel('iteration')
        plt.ylabel('reward')
        plt.savefig('./figure/q2.png')
        plt.close()
        
