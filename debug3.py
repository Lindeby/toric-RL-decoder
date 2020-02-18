from src.ToricCode import ToricCode
from src.actor import selectAction, generateTransition
from src.nn.torch.ResNet import ResNet18

import objgraph




def worker(T):
    config =    {  "size": 9,
                "min_qubit_errors": 0,
                "p_error": 0.5
                }
                 
    env = ToricCode(config=config)
    model = ResNet18()

    state = env.reset()

    objgraph.show_growth(limit=5)

    for t in range(T):

        a, _ = selectAction(
            number_of_actions = 3,
            epsilon           = 0.4,
            grid_shift        = int(env.system_size/2),
            toric_size        = env.system_size,
            state             = state,
            model             = model,
            device            = 'cpu'
        )

        next_state, reward, terminal, _ = env.step(a)

        transition = generateTransition(a, reward, int(env.system_size/2), state, next_state, terminal)

        if t > 10:
            objgraph.show_growth()


if __name__ == "__main__":
    worker(100)

