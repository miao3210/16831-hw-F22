import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt 
import numpy as np 

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

def sort_lambda(paths):
    lambdas = [p.split('lam')[1].split('_')[0] for p in paths]
    lambdas = [float(lam) for lam in lambdas]
    lambdas = np.array(lambdas)
    ind = np.argsort(lambdas)
    return np.array(paths)[ind]


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    #args = parser.parse_args()

    pathsold = [
        'hw5_expl_q2_rwr_easy_lam0.1_PointmassEasy-v0_01-12-2022_20-59-35',
        'hw5_expl_q2_rwr_easy_lam1_PointmassEasy-v0_01-12-2022_17-32-52', 
        'hw5_expl_q2_rwr_easy_lam2_PointmassEasy-v0_01-12-2022_17-32-52', 
        'hw5_expl_q2_rwr_easy_lam10_PointmassEasy-v0_01-12-2022_17-32-53',
        'hw5_expl_q2_rwr_easy_lam20_PointmassEasy-v0_01-12-2022_17-32-52',
        'hw5_expl_q2_rwr_easy_lam50_PointmassEasy-v0_01-12-2022_17-32-53',
        #'hw5_expl_q2_rwr_medium_lam0.1_PointmassMedium-v0_01-12-2022_16-48-37',
        'hw5_expl_q2_rwr_medium_lam0.1_PointmassMedium-v0_01-12-2022_17-06-56',
        #'hw5_expl_q2_rwr_medium_lam0.1_PointmassMedium-v0_01-12-2022_21-19-51',
        'hw5_expl_q2_rwr_medium_lam1_PointmassMedium-v0_01-12-2022_17-32-53',  
        'hw5_expl_q2_rwr_medium_lam2_PointmassMedium-v0_01-12-2022_17-32-52',  
        #'hw5_expl_q2_rwr_medium_lam2_PointmassMedium-v0_01-12-2022_21-19-50',
        #'hw5_expl_q2_rwr_medium_lam0_PointmassMedium-v0_01-12-2022_16-29-56',  
        #'hw5_expl_q2_rwr_medium_lam0_PointmassMedium-v0_01-12-2022_16-40-34',  
        'hw5_expl_q2_rwr_medium_lam10_PointmassMedium-v0_01-12-2022_17-32-52', 
        'hw5_expl_q2_rwr_medium_lam20_PointmassMedium-v0_01-12-2022_17-32-53', 
        'hw5_expl_q2_rwr_medium_lam50_PointmassMedium-v0_01-12-2022_17-32-53'
        ]
    paths = [
        'hw5_expl_q2_rwr_medium_lam10_PointmassMedium-v0_02-12-2022_14-41-41',
        'hw5_expl_q2_rwr_easy_lam1_PointmassEasy-v0_02-12-2022_14-41-41',
        'hw5_expl_q2_rwr_easy_lam20_PointmassEasy-v0_02-12-2022_14-41-41',
        'hw5_expl_q2_rwr_medium_lam1_PointmassMedium-v0_02-12-2022_14-41-42',
        'hw5_expl_q2_rwr_easy_lam50_PointmassEasy-v0_02-12-2022_14-41-42',
        'hw5_expl_q2_rwr_medium_lam0.1_PointmassMedium-v0_02-12-2022_14-41-40',
        'hw5_expl_q2_rwr_easy_lam2_PointmassEasy-v0_02-12-2022_14-41-40',
        'hw5_expl_q2_rwr_medium_lam2_PointmassMedium-v0_02-12-2022_14-41-41',
        'hw5_expl_q2_rwr_medium_lam50_PointmassMedium-v0_02-12-2022_14-41-40',
        'hw5_expl_q2_rwr_easy_lam0.1_PointmassEasy-v0_02-12-2022_14-41-42',
        'hw5_expl_q2_rwr_medium_lam20_PointmassMedium-v0_02-12-2022_14-41-41',
        'hw5_expl_q2_rwr_easy_lam10_PointmassEasy-v0_02-12-2022_14-41-41',
    ]
    paths = ['../../data/' + p for p in paths]
    paths = sort_lambda(paths)
    easy = []
    medium = []
    for p in paths:
        if 'easy' in p:
            easy.append(p)
        elif 'medium' in p:
            medium.append(p)

    plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('evaluation average return')
    plt.title('q2 RWR easy')
    #plt.xlim([0,70000])
    for p in easy:
        logdir = os.path.join(p, 'events*')
        print(logdir)
        label = '$\lambda$ = ' +  logdir.split('lam')[1].split('_')[0]
        eventfile = glob.glob(logdir)[0]

        X, Y, stepX, stepY = get_section_results(eventfile)
        if stepY[-1] < 4e4:
            print(logdir, stepY[-1])
            continue
        plt.plot(stepY, Y, label=label)
        #for i, (x, y) in enumerate(zip(X, Y)):
        #    print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
    plt.legend(loc='lower right')
    plt.savefig('./figure/q2_RWR_easy.png')
    
    plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('evaluation average return')
    plt.title('q2 RWR medium')
    #plt.xlim([0,70000])
    for p in medium:
        logdir = os.path.join(p, 'events*')
        print(logdir)
        label = '$\lambda$ = ' +  logdir.split('lam')[1].split('_')[0]
        eventfile = glob.glob(logdir)[0]

        X, Y, stepX, stepY = get_section_results(eventfile)
        if stepY[-1] < 4e4:
            print(logdir, stepY[-1])
            continue
        plt.plot(stepY, Y, label=label)
        #for i, (x, y) in enumerate(zip(X, Y)):
        #    print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
    plt.legend(loc='upper left')
    plt.savefig('./figure/q2_RWR_medium.png')
        
