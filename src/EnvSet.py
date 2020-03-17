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


    def resetTerminalEnvs(self, idx, p_errors=None):
        states = np.empty((len(idx), 2, self.size, self.size))
        if p_errors is None:
            for i, id in enumerate(idx):
                states[i] = self.envs[id].reset()
        else:
            for i, id in enumerate(idx):
                states[i] = self.envs[id].reset(p_error=p_errors[i])
        return states

    def resetAll(self, p_errors=None):
        if p_errors is None:
            for i, e in enumerate(self.envs):
                self.states[i] = e.reset()
        else:
            for i, e in enumerate(self.envs):
                self.states[i] = e.reset(p_error=p_errors[i])
        return self.states

    def step(self, actions):
        states     = np.empty((self.no_envs, 2, self.size, self.size), dtype=np.int)
        rewards    = np.empty( self.no_envs)
        terminals  = np.zeros( self.no_envs, dtype=np.bool)
        for i, e in enumerate(self.envs):
            next_state, reward, terminal, info = e.step(actions[i])
            states[i]    = next_state
            rewards[i]   = reward
            terminals[i] = terminal
        return states, rewards, terminals, info


    def isAnyTerminal(self):
        return not (len(self._terminal_envs) == 0)
