import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt 

def get_section_results_original(file):
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

    paths = [
        'hw5_expl_q1_hard_dqn_transform_PointmassHard-v0_03-12-2022_17-56-44',
        'hw5_expl_q1_hard_cql_transform_PointmassHard-v0_03-12-2022_18-33-03',
        'hw5_expl_q1_medium_dqn_transform_PointmassMedium-v0_03-12-2022_19-39-39',
        'hw5_expl_q1_medium_cql_transform_PointmassMedium-v0_03-12-2022_19-07-06',
        #'hw5_expl_q1_hard_cql_transform_PointmassHard-v0_03-12-2022_17-38-28',
        #'hw5_expl_q1_medium_cql_PointmassMedium-v0_03-12-2022_17-17-18',
        #'hw5_expl_q1_hard_dqn_transform_PointmassHard-v0_03-12-2022_17-48-50',
        #'hw5_expl_q1_hard_cql_PointmassHard-v0_03-12-2022_17-17-19',
        #'hw5_expl_q1_medium_dqn_PointmassMedium-v0_03-12-2022_17-17-18',
        #'hw5_expl_q1_hard_dqn_PointmassHard-v0_03-12-2022_17-17-18', too short
        #'hw5_expl_q1_hard_cql_transform_PointmassHard-v0_03-12-2022_17-34-32',
        #'hw5_expl_q1_hard_cql_transform_PointmassHard-v0_03-12-2022_17-43-15',

        'hw5_expl_q1_hard_dqn_PointmassHard-v0_02-12-2022_19-43-16',
        #'hw5_expl_q1_hard_cql_PointmassHard-v0_02-12-2022_19-43-16',
        'hw5_expl_q1_medium_dqn_PointmassMedium-v0_02-12-2022_19-43-15',
        #'hw5_expl_q1_medium_cql_PointmassMedium-v0_02-12-2022_19-43-16',

        'hw5_expl_q1_medium_cql_PointmassMedium-v0_03-12-2022_20-41-59',
        'hw5_expl_q1_hard_cql_PointmassHard-v0_03-12-2022_21-13-47',
    ]
    paths = ['../../data/' + p for p in paths]

    hard = []
    medium = []
    for p in paths:
        if 'hard' in p:
            hard.append(p)
        elif 'medium' in p:
            medium.append(p)
    
    plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('evaluation average return')
    plt.title('q1 DQN vs CQL hard')
    for p in hard:
        logdir = os.path.join(p, 'events*')
        print(logdir)
        label = logdir.split('hard_')[1].split('_Point')[0]
        eventfile = glob.glob(logdir)[0]

        X, Y, stepX, stepY = get_section_results(eventfile)
        plt.plot(stepY, Y, '*-', label=label)
        #for i, (x, y) in enumerate(zip(X, Y)):
        #    print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
    plt.legend(loc='upper left')
    plt.savefig('./figure/q1_CQL_hard.png')
    
    plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('evaluation average return')
    plt.title('q1 DQN vs CQL medium')
    for p in medium:
        logdir = os.path.join(p, 'events*')
        print(logdir)
        label = logdir.split('medium_')[1].split('_P')[0]
        eventfile = glob.glob(logdir)[0]

        X, Y, stepX, stepY = get_section_results(eventfile)
        plt.plot(stepY, Y, '*-', label=label)
        #for i, (x, y) in enumerate(zip(X, Y)):
        #    print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
    plt.legend(loc='upper left')
    plt.savefig('./figure/q1_CQL_medium.png')
        
