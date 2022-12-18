import abc
from argparse import Action
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
            # miao
            self.cross_entropy = nn.CrossEntropyLoss()
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )
            # miao
            self.gaussiannll = nn.GaussianNLLLoss() #input#prob, target#label, var) or distribution.log_likelihood(action)

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        #raise NotImplementedError
        # miao
        observation = ptu.from_numpy(observation).to(ptu.device)
        action = self.forward(observation) 
        # import pdb
        # pdb.set_trace()
        scale_factor = 0.0001
        action += self.logstd.exp() * torch.normal(0*action.detach(), 0*action.detach()+scale_factor)
        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError
        
    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        #raise NotImplementedError
        # miao
        mean = self.mean_net(observation)
        #prob = torch.distributions.normal.Normal(loc=mean, scale=self.logstd.exp())
        return mean



#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss_mse = nn.MSELoss()
        self.iterations = 0

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
        
        #loss = TODO
        # miao
        self.optimizer.zero_grad()
        
        observations = ptu.from_numpy(observations).to(ptu.device)
        action_predict = self.forward(observations)
        #NLL or MSE
        #loss = nn.GaussianNLLLoss(input#prob, target#label, var) or distribution.log_likelihood(action)
        #loss_value = self.gaussiannll(mean, actions, std)
        # import pdb 
        # pdb.set_trace()
        actions = ptu.from_numpy(actions).to(ptu.device)
        loss = self.loss_mse(actions, action_predict)
        loss.backward()
        self.optimizer.step()
        #print('iteration={}, loss: {}'.format(self.iterations, loss))
        self.iterations += 1

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
            # miao
            #'Training mean': ptu.to_numpy(mean),
            #'action': actions
        }
