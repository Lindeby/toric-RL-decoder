import gym, gym_ToricCode
from src.nn.torch.ResNet import ResNet18
from src.learner import predictMax, predictMaxOptimized
import numpy as np
import torch, time

if __name__ == "__main__":

    env_config = {"size":9, "min_qbit_errors":0, "p_error":0.1}
    env = gym.make('toric-code-v0', config=env_config)
    model = ResNet18()

    model.eval()
    for _ in range(100):
        state = env.reset()

        batch = np.array([np.zeros((2,9,9)), state, state, state, state])

        start = time.time()
        t1 = predictMax(model, batch, len(batch), int(9/2), 9, 'cpu')
        end0 = time.time()
        t2 = predictMaxOptimized(model, batch, int(9/2), 9, 'cpu')
        end1 = time.time()

        print("reg", end0-start)
        print("opt", end1-end0)
        print(torch.all(t1.eq(t2)))
