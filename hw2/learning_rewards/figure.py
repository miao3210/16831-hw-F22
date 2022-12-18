import matplotlib.pyplot as plt 
import numpy as np 
import os 
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--direc', type=str, default='./')
parser.add_argument('--figname', type=str)
parser.add_argument('--title', type=str)
parser.add_argument('--cond', type=str)
parser.add_argument('--excep', default='none', type=str)
parser.add_argument('--loc', default='upper left', type=str)
parser.add_argument('--clip', default='', type=str)

args = parser.parse_args()
params = vars(args)
figname = params['figname']
title = params['title'].replace('+', ' ')
cond = params['cond'].split('+')
excep = params['excep'].split('+')
direc = params['direc']
loc = params['loc'].replace('+', ' ')
clip = params['clip']
print(direc)
#direc = './'
#figname = 'q4-2.png' #'q2_lr_b500.png'#'q4.png' #'q5_lambda_095.png' #'q2_seed_4_lr.png'
#title = 'HalfCheetah-v4: Reward-to-go and NN-Baseline' #'InvertedPendulum-v4: Search for the Largest lr with b=500'# Smallest b'# and the Largest lr' #'HalfCheetah-v4: Search for Batchsize and Learning Rate' #'LunarLanderContinuous-v4 Learning Curve' #'Hopper-v4: GAE with lambda=0.95'
#cond = ['q2_b_5000', 'eval']
#cond = ['q4_b_', 'eval'] #['q2_b_500_', 'eval']#['q4', 'search_b', 'eval']#['q3', 'eval'] #['q5', 'lambda_0.95', 'eval']
#excep = ['seed'] #['lambda_0_']

def check(cond, expname):
    label = True
    for c in cond:
        label = label * (c in expname)
    for e in excep:
        label = label * (not e in expname)
    return label

def plt_fill_between(expname, ms, pos):
    mean = ms[:, 0]
    std = ms[:, 1]
    up = mean + std/2
    low = mean - std/2
    x = np.array(range(mean.shape[0]))
    # for q2
    #if mean.max() < 999:
    #    return 0
    if pos > 0:
        plt.subplot(3,3, pos) #(2, 5, pos)
    plt.fill_between(x, low, up, alpha=0.3)
    if clip == '': 
        plt.plot(x, mean, label=expname)# for q4 .split('_rtg')[0].split('search_')[1])
    else:
        plt.plot(x, mean, label=expname.split(clip)[0])
    #plt.legend(loc='upper')
    #plt.title(expname)
    #plt.xlabel('iteration')
    #plt.ylabel('evaluation rewards')

names = []
curves = []
txtfiles = os.listdir(direc)
txtfiles = sorted(txtfiles)
plt.figure(figsize=(12,5))#(10,6)) #(21,15))#(10,5)
pos = 0
txtfiles = [k*check(cond, k) for k in txtfiles]
for f in txtfiles:
    if f.endswith('.txt'):
        if len(f) > 4: #check(cond, f):
            #print(f)
            #import pdb 
            #pdb.set_trace()
            names.append(f.split('.')[0])
            curves.append(np.loadtxt(direc+f))
            plt_fill_between(f[:-4], np.loadtxt(direc+f), pos)
            #pos += 1
plt.legend(loc=loc)
plt.xlabel('iteration')
plt.ylabel('evaluation rewards')
plt.title(title)
plt.savefig(figname, bbox_inches='tight')

#import pdb
#pdb.set_trace()
