import copy
import numpy as np

class EnvSet():
    def __init__(self, env, no_envs):

        self.states        = np.empty((no_envs, 2, env.system_size, env.system_size))
        self.rewards       = np.empty( no_envs)
        self.terminals     = np.zeros( no_envs, dtype=np.bool)


        self.envs = []
        for i in range(no_envs):
            self.envs.append(copy.deepcopy(env))


    def resetTerminalEnvs(self, idx):
        for id in idx:
            self.states[id] = self.envs[id].reset()
        return self.states

    def resetAll(self):
        for i, e in enumerate(self.envs):
            self.states[i] = e.reset()
        return self.states

    def step(self, actions):
        for i, e in enumerate(self.envs):
            next_state, reward, terminal, _ = e.step(actions[i])
            self.states[i]    = next_state
            self.rewards[i]   = reward
            self.terminals[i] = terminal
            if terminal:
                self._terminal_envs.append(i)
        return self.states, self.rewards, self.terminals, {}


    def isAnyTerminal(self):
        return not (len(self._terminal_envs) == 0)
