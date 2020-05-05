'''
AUTHOR : SWASTIK NATH

Copyright 2020 SWASTIK NATH.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.policy_s = np.ones(self.nA)

    def select_action(self, state, i_episode, eps=None):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
        self.policy_s = np.ones(self.nA) * epsilon / self.nA
        best_a = np.argmax(self.Q[state])
        self.policy_s[best_a] = 1 - epsilon + (epsilon / self.nA)
        
        action_to_select = np.random.choice(np.arange(self.nA), p=self.policy_s)
        return action_to_select

    def step(self, state, action, reward, next_state, done, alpha=0.02, gamma=0.7):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        old_q = self.Q[state][action]
        next_action = np.max(self.Q[next_state])
        self.Q[state][action] += alpha * (reward + (gamma * next_action) - old_q)
        