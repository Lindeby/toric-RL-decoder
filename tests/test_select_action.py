from src.util_actor import selectAction, selectActionParallel
from src.nn.torch.ResNet import ResNet18
import gym, gym_ToricCode, torch
import numpy as np


if __name__ == "__main__":
    env_config = {  
        "size": 3,
        "min_qubit_errors": 0,
        "p_error": 0.1
    }

    env = gym.make('toric-code-v0', config=env_config)
    states = []

    for i in range(1000):
        states.append(env.reset())

    states = np.array(states)

    model = ResNet18()

    actions1 = []
    for state in states:
        actions1.append(selectAction(3, 0, 1, 3, state, model, torch.device('cpu')))

    action2, qv2 = selectActionParallel(3,0,1,3, states, model, torch.device('cpu'))

    for i in range(len(actions1)):
        a1 = actions1[i][0]
        q1 = actions1[i][1]
        a2 = action2[i] 
        q2 = qv2[i]

        if not (np.all(q1-q2 < 1e-8) and np.all(a1 == a2)):
            print(np.all(a1 == a2), a1, a2)
            print(np.all(np.equal(q1,q2)), q1,q2)
            exit()
        
        

