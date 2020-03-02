import copy
import numpy as np

class EnvSet():
    def __init__(self, env, no_envs):

        self.size       = env.system_size
        self.no_envs    = no_envs
        self.states     = np.empty((no_envs, 2, self.size, self.size), dtype=np.int)
        self.rewards    = np.empty( no_envs)
        self.terminals  = np.zeros( no_envs, dtype=np.bool)


        self.envs = []
        for i in range(no_envs):
            self.envs.append(copy.deepcopy(env))


    def resetTerminalEnvs(self, idx):
        states = np.empty((len(idx), 2, self.size, self.size))
        for i, id in enumerate(idx):
            states[i] = self.envs[id].reset()
        return states

    def resetAll(self):
        for i, e in enumerate(self.envs):
            self.states[i] = e.reset()
        return self.states

    def step(self, actions):
        states     = np.empty((self.no_envs, 2, self.size, self.size), dtype=np.int)
        rewards    = np.empty( self.no_envs)
        terminals  = np.zeros( self.no_envs, dtype=np.bool)
        for i, e in enumerate(self.envs):
            next_state, reward, terminal, _ = e.step(actions[i])
            states[i]    = next_state
            rewards[i]   = reward
            terminals[i] = terminal
        return states, rewards, terminals, {}


    def isAnyTerminal(self):
        return not (len(self._terminal_envs) == 0)
