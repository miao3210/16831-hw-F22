import numpy as np 
import time 

rewards = np.random.randn(1000)
class veri(object):
    def __init__(self,):
        self.gamma = 0.5

    def v(self, rewards):
        T = len(rewards)
        discounted_cumsum = np.array(rewards)
        order = np.array(range(T))
        discount = self.gamma ** order
        discounted_cumsum = discounted_cumsum * discount 
        discounted_cumsum = np.flip(np.flip(discounted_cumsum).cumsum())
        causal = self.gamma ** (-order)
        discounted_cumsum = discounted_cumsum * causal
        list_of_discounted_cumsums = discounted_cumsum.tolist()
        
        return list_of_discounted_cumsums 

    def loo(self, rewards):
        T = len(rewards)
        discounted_cumsum = np.zeros(T)
        for t in range(T):
            for tp in range(t, T):
                tmp = self.gamma ** (tp - t) * rewards[tp]
                discounted_cumsum[t] += tmp
        list_of_discounted_cumsums = discounted_cumsum.tolist()
        
        return list_of_discounted_cumsums 

verify = veri()
t0 = time.time()
vec = verify.v(rewards) 
t1 = time.time()
loo = verify.loo(rewards)
t2 = time.time()
print([vec[-1], t1-t0])
print([loo[-1], t2-t1])