from matplotlib import pyplot as plt 
import numpy as np 

env = 'halfcheetah' #'walker2d' #'ant'
scalefactor = '0' #'1e-4' # '0'
q2ant = './hw1/' + env + scalefactor + '.txt'
meanstd = np.loadtxt(q2ant)
mean = meanstd[0, :]
std = meanstd[1, :]
below = mean - std/2
up = mean + std/2
x = np.arange(10)
plt.fill_between(x, up, below, alpha=0.3)
plt.plot(x, mean, '>-', label='DAgger')
expertmean = 4813.9560546875*np.ones(10)
expertstd = 72.0189208984375*np.ones(10)
plt.fill_between(x, (expertmean - expertstd/2), (expertmean + expertstd/2), alpha=0.3)
plt.plot(x, expertmean, '<-', label='Expert')
plt.legend(loc='lower right')
plt.title('DAgger on ' + env + ' with scale factor='+scalefactor)
plt.xlabel('iteration')
plt.ylabel('reward of validation')
plt.savefig(env + scalefactor + '.png')

quit()

q13 = './hw1/q13sorted.txt'
qq = np.loadtxt(q13)

below = qq[:, 1] - qq[:, 2]/2 
up = qq[:, 1] + qq[:, 2]/2
mid = qq[:, 1]
x = -np.log10(qq[:,0] + 1e-21)
plt.fill_between(x, below, up, alpha=0.3)
plt.plot(x, mid, '>-')

plt.title('Ant-v4: scale factor and the trade-off of exploitation & exploration')
plt.xlabel('-log10(scale factor + \epsilon)')
plt.ylabel('reward of evaluation')

plt.savefig('q1.png')