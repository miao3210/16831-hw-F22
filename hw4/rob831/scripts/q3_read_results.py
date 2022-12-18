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

    paths = ['hw4_q3_cheetah_cheetah-rob831-v0_15-11-2022_21-49-34/', 
        'hw4_q3_obstacles_obstacles-rob831-v0_15-11-2022_01-49-51/',
        'hw4_q3_reacher_reacher-rob831-v0_16-11-2022_00-33-35/']
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
        
        label = p.split('q3_')[1].split('_')[0]
        #plt.plot(stepX, X, label=label + ' train average reward')
        plt.plot(stepY, Y, label=label + ' eval average reward')
        plt.legend()
        plt.title('q3: ' + label)
        plt.xlabel('iteration')
        plt.ylabel('reward')
        plt.savefig('./figure/q3_'+label+'.png')
        plt.close()
        
