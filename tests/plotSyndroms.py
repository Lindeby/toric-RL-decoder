import gym, gym_ToricCode

if __name__ == "__main__":
    env_config = {  
        "size": 5,
        "min_qubit_errors": 0,
        "p_error": 0.1
    }

    env = gym.make('toric-code-v0', config=env_config)
    state = env.reset()
    env.plotToricCode(state, "plotting")