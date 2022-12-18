import argparse
import glob
import os
import tensorflow.compat.v1 as tf

## miao
import pdb 
import matplotlib.pyplot as plt 


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
    ../../data/q1_MsPacman-v0_30-10-2022_16-00-57
    '''
    #args = parser.parse_args()
    logdir = '../../data/q1_MsPacman-v0_30-10-2022_16-00-57'
    logdir = os.path.join(logdir, 'events*')
    print('logdir={}'.format(logdir))
    eventfile = glob.glob(logdir)[0]

    X, Y, Z, stepX, stepY, stepZ = get_section_results(eventfile)
    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        pass #print('Iteration {:d} | Train steps: {:d} | Return: {} | Best Return: {}'.format(i, int(x), y, z))
    #pdb.set_trace()
    print(len(stepY))
    plt.figure(figsize=(10,8))
    plt.plot(stepY, [1500]*len(stepY), '--', label='1500')
    plt.plot([1e6, 1e6], [400, 1700], '--', label='t=1e6')
    plt.plot(stepY, Y, label='average reward')
    plt.plot(stepZ, Z, label='best reward')
    plt.xlabel('time step')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend(loc = 'upper left')
    plt.ylabel('reward')
    plt.title(args.logdir.split('/')[-1])
    plt.savefig('./figures/' + args.logdir.split('/')[-1]+'.png', bbox_inches='tight')