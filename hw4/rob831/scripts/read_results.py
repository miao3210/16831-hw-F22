import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #../../data/hw4_q2_obstacles_singleiteration_obstacles-rob831-v0_14-11-2022_20-17-11
    #../../data/hw4_q3_cheetah_cheetah-rob831-v0_14-11-2022_20-05-31
    #../../data/hw4_q3_reacher_reacher-rob831-v0_14-11-2022_20-08-28
    #../../data/hw4_q4_reacher_numseq100_reacher-rob831-v0_14-11-2022_20-38-25
    parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    args = parser.parse_args()

    logdir = os.path.join(args.logdir, 'events*')
    print(logdir)
    eventfile = glob.glob(logdir)[0]

    X, Y = get_section_results(eventfile)
    print(X)
    print(Y)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
