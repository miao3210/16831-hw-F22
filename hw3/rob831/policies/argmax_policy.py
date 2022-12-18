import numpy as np

class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        ## miao
        qa_values = self.critic.qa_values(observation) # shape (1,9)
        action = np.argmax(qa_values, axis=-1)
        #import pdb 
        #pdb.set_trace()
        #print(qa_values.shape)
        return action.squeeze()

    def get_exploration(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        qa_values = self.critic.qa_values(observation)

        # hparam1: use qa as distribution
        if True:
            psoftmax = np.exp(qa_values) / np.sum(np.exp(qa_values))
            action = np.random.Generator.choice(qa_values.shape[1], p=psoftmax)    
        # hparam2: use min 
        if False:
            action = np.argmin(qa_values, axis=-1)
        # hparam3: use second max
        if False:
            amax = np.argmax(qa_values, axis=-1)
            qa_values[amax] = -1e10
            action = np.argmax(qa_values, axis=-1)
        return action.squeeze()